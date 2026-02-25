import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# =========================
# CONFIG
# =========================
DATASET_DIR = "data_main"
MODEL_PATH = "model.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORM (NO AUGMENTATION)
# =========================
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# LOAD DATASET
# =========================
dataset = datasets.ImageFolder(DATASET_DIR, transform=eval_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# =========================
# DEFINE MODEL (MUST MATCH TRAINING)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),           # helps regularize
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = SimpleCNN(num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================
# INFERENCE
# =========================
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# =========================
# ROC CURVE
# =========================
if num_classes == 2:
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

else:
    # Multi-class ROC
    labels_bin = label_binarize(all_labels, classes=range(num_classes))

    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Multi-class ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

# =========================
# PER-CLASS F1 BAR CHART
# =========================
report = classification_report(
    all_labels, all_preds,
    target_names=class_names,
    output_dict=True
)

f1_scores = [report[c]["f1-score"] for c in class_names]

plt.figure(figsize=(8, 5))
plt.bar(class_names, f1_scores)
plt.title("Per-Class F1 Scores")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.show()

# =========================
# MISCLASSIFIED EXAMPLES
# =========================
print("\nShowing some misclassified examples...")

misclassified = np.where(all_preds != all_labels)[0][:9]

plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified):
    img, label = dataset[idx]
    img = img.permute(1, 2, 0).numpy()
    img = (img * 0.229 + 0.485)  # approximate un-normalization

    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(f"True: {class_names[all_labels[idx]]}\nPred: {class_names[all_preds[idx]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()