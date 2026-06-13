import torch
import torchvision.models as models
import onnx
import tensorflow as tf
import os

checkpoint = torch.load('model/original.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']
all_keys = "".join(state_dict.keys())

is_v3_small = any("features.1.block" in key for key in state_dict.keys())
is_v3_large = any("features.13.block" in key for key in state_dict.keys())

def initialize_model(num_classes):
    if is_v3_small:
        model = models.mobilenet_v3_small()
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    elif is_v3_large:
        model = models.mobilenet_v3_large()
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        model = models.mobilenet_v2()
        model.classifier[1] = torch.nn.Linear(1280, num_classes)
    return model

try:
    model = initialize_model(10)
    model.load_state_dict(state_dict, strict=True)
except RuntimeError as e:
    if "size mismatch for classifier" in str(e) or "Unexpected key(s) in state_dict" in str(e):
        model = initialize_model(9)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise e

model.eval()

# 1. Konversi ke ONNX
print("1. Mengonversi PyTorch ke ONNX...")
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
print("Berhasil menyimpan model.onnx")

# 2. Konversi ONNX ke TensorFlow SavedModel
print("\n2. Mengonversi ONNX ke TensorFlow SavedModel...")
tf_model_path = "saved_model_tf"
os.system(f"onnx2tf -i {onnx_path} -o {tf_model_path}")

# 3. Konversi TensorFlow SavedModel ke TFLite
print("\n3. Mengonversi TensorFlow ke TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

# Opsional: Optimasi/Kuantisasi agar model super ringan di Raspberry Pi
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_path = "model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"\nSelesai! File sukses dikonversi menjadi: {tflite_path}")