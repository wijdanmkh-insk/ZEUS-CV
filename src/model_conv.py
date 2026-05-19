from ultralytics import YOLO
import shutil

# Muat model .pt hasil training kamu
model = YOLO("ref/best.pt")

# Ekspor model ke format ONNX
model.export(format="onnx", name="../model/best.onnx")