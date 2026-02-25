import torch
import torch.nn as nn

# =========================
# CONFIG
# =========================
MODEL_PATH = "model.pth"
ONNX_PATH = "model.onnx"
IMG_SIZE = 128   # must match training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL DEFINITION
# MUST MATCH TRAINING EXACTLY
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
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # CHANGE if num_classes different
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =========================
# LOAD MODEL
# =========================
num_classes = 3  # CHANGE if needed
model = SimpleCNN(num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")

# =========================
# EXPORT TO ONNX
# =========================
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print(f"Model successfully exported to {ONNX_PATH}")