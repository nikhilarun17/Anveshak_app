import onnxruntime as ort
import numpy as np
import cv2

# =========================
# CONFIG
# =========================
ONNX_PATH = "model.onnx"
IMG_SIZE = 128
CLASS_NAMES = ["blue", "green", "orange"]  # must match training

# =========================
# LOAD MODEL
# =========================
session = ort.InferenceSession(ONNX_PATH)

# =========================
# LOAD IMAGE
# =========================
image_path = "test.png"  # change this
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# Normalize (must match training!)
img = img.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img = (img - mean) / std

# Convert to CHW format
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0).astype(np.float32)

# =========================
# RUN INFERENCE
# =========================
outputs = session.run(None, {"input": img})
pred = np.argmax(outputs[0])

print("Predicted class:", CLASS_NAMES[pred])