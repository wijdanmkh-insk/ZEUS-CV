import time
import torch
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from collections import Counter

CAMERA_INDEX = 1

class_names = {
    0: "Background", 1: "Cardboard", 2: "Food Organics", 3: "Glass", 4: "Metal",
    5: "Miscellaneous Trash", 6: "Paper", 7: "Plastic", 8: "Textile Trash", 9: "Vegetation"
}

checkpoint = torch.load('original.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']

is_v3_small = any("features.1.block" in key for key in state_dict.keys())
is_v3_large = any("features.13.block" in key for key in state_dict.keys())

def initialize_model(num_classes):
    if is_v3_small:
        model = models.mobilenet_v3_small()
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        model_name = "MobileNetV3 Small"
    elif is_v3_large:
        model = models.mobilenet_v3_large()
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        model_name = "MobileNetV3 Large"
    else:
        model = models.mobilenet_v2()
        model.classifier[1] = torch.nn.Linear(1280, num_classes)
        model_name = "MobileNetV2"
    return model, model_name

try:
    model, detected_name = initialize_model(10)
    model.load_state_dict(state_dict, strict=True)
    detected_classes = 10
except RuntimeError as e:
    if "size mismatch for classifier" in str(e) or "Unexpected key(s) in state_dict" in str(e):
        model, detected_name = initialize_model(9)
        model.load_state_dict(state_dict, strict=False)
        detected_classes = 9
    else:
        raise e

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Model Terdeteksi     : {detected_name}")
print(f"Jumlah Kelas         : {detected_classes}")
print(f"Device yang digunakan: {device.type.upper()}")
print(f"Membuka Kamera Indeks: {CAMERA_INDEX}\n")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Kamera indeks {CAMERA_INDEX} tidak dapat diakses.")
    exit()

detection_buffer = []
latencies = []
start_timer = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil gambar dari kamera.")
            break

        input_tensor = transform(frame).unsqueeze(0).to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        
        inference_time = (t_end - t_start) * 1000
        latencies.append(inference_time)

        _, predicted = torch.max(output, 1)
        pred_class = predicted.item()
        detection_buffer.append(pred_class)

        elapsed_time = time.time() - start_timer

        if elapsed_time >= 3.0:
            if detection_buffer:
                most_common_class = Counter(detection_buffer).most_common(1)[0][0]
                final_display_text = class_names.get(most_common_class, f"Class {most_common_class}")
                avg_latency = sum(latencies) / len(latencies)
                print(f"[HASIL 3s] Terbanyak: {final_display_text:<20} | Rata-rata Latensi: {avg_latency:.2f} ms")
            detection_buffer = []
            latencies = []
            start_timer = time.time()

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nProgram Headless dihentikan.")

cap.release()