import time
import torch
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from collections import Counter

FRAME_CAP = 0

class_names = {
    0: "Background", 1: "Cardboard", 2: "Food Organics", 3: "Glass", 4: "Metal",
    5: "Miscellaneous Trash", 6: "Paper", 7: "Plastic", 8: "Textile Trash", 9: "Vegetation"
}

checkpoint = torch.load('model/finalmodel.pth', map_location='cpu')
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

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(FRAME_CAP, cv2.CAP_V4L2)
detection_buffer = []
latencies = []
start_timer = time.time()
final_display_text = "Menghitung..."
inference_text = "Inference: 0.00 ms"

while True:
    ret, frame = cap.read()
    if not ret:
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
    
    inf_time = (t_end - t_start) * 1000
    latencies.append(inf_time)
    inference_text = f"Inference: {inf_time:.2f} ms"

    _, predicted = torch.max(output, 1)
    pred_class = predicted.item()
    detection_buffer.append(pred_class)

    elapsed_time = time.time() - start_timer
    countdown = max(0, 3.0 - elapsed_time)

    if elapsed_time >= 3.0:
        if detection_buffer:
            most_common_class = Counter(detection_buffer).most_common(1)[0][0]
            final_display_text = class_names.get(most_common_class, f"Class {most_common_class}")
            avg_latency = sum(latencies) / len(latencies)
            print(f"[LOG 3s] Terbanyak: {final_display_text:<20} | Rata-rata Latensi: {avg_latency:.2f} ms")
        detection_buffer = []
        latencies = []
        start_timer = time.time()

    cv2.putText(frame, f"Objek (3s Avg): {final_display_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, inference_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Update dalam: {countdown:.1f}s", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('ZEUS CV - Realtime Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()