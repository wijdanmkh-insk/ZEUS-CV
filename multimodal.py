import time
import torch
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from collections import Counter

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.85

class_names_A = {0: "Background", 2: "Food Organics", 9: "Vegetation"}  
class_names_B = {1: "Cardboard", 3: "Glass", 4: "Metal", 5: "Miscellaneous Trash", 6: "Paper", 7: "Plastic", 8: "Textile Trash"}

def load_dynamic_model(path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    all_keys = "".join(state_dict.keys())
    
    is_v3_small = any("features.1.block" in key for key in state_dict.keys())
    is_v3_large = any("features.13.block" in key for key in state_dict.keys())
    
    try:
        classes = 10
        if is_v3_small:
            model = models.mobilenet_v3_small()
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 10)
        elif is_v3_large:
            model = models.mobilenet_v3_large()
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 10)
        else:
            model = models.mobilenet_v2()
            model.classifier[1] = torch.nn.Linear(1280, 10)
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        classes = 9
        if is_v3_small:
            model = models.mobilenet_v3_small()
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 9)
        elif is_v3_large:
            model = models.mobilenet_v3_large()
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 9)
        else:
            model = models.mobilenet_v2()
            model.classifier[1] = torch.nn.Linear(1280, 9)
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    return model

model_v3 = load_dynamic_model('mv3.pth')
model_v2 = load_dynamic_model('mv2.pth')

device = torch.device('cpu')
model_v3.to(device)
model_v2.to(device)

print("Sistem Multi-Model Aktif (MobileNetV3 & MobileNetV2)")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    exit()

detection_buffer = []
latencies = []
start_timer = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = transform(frame).unsqueeze(0).to(device)
        t_start = time.perf_counter()

        with torch.no_grad():
            output_v3 = model_v3(input_tensor)
            probs_v3 = torch.nn.functional.softmax(output_v3, dim=1)
            conf_v3, pred_v3 = torch.max(probs_v3, 1)
            
            val_conf_v3 = conf_v3.item()
            val_pred_v3 = pred_v3.item()

            if val_conf_v3 >= CONFIDENCE_THRESHOLD and val_pred_v3 in class_names_A:
                final_pred = val_pred_v3
                final_source = "MobileNetV3 (Spesialis)"
            else:
                output_v2 = model_v2(input_tensor)
                probs_v2 = torch.nn.functional.softmax(output_v2, dim=1)
                _, pred_v2 = torch.max(probs_v2, 1)
                final_pred = pred_v2.item()
                final_source = "MobileNetV2 (Fallback)"

        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)
        detection_buffer.append(final_pred)

        elapsed_time = time.time() - start_timer
        if elapsed_time >= 3.0:
            if detection_buffer:
                most_common_class = Counter(detection_buffer).most_common(1)[0][0]
                
                if most_common_class in class_names_A:
                    name_trash = class_names_A[most_common_class]
                else:
                    name_trash = class_names_B.get(most_common_class, f"Class {most_common_class}")
                    
                avg_latency = sum(latencies) / len(latencies)
                print(f"[ZEUS COMBO] Terbanyak: {name_trash:<20} | Rata-rata Latensi: {avg_latency:.2f} ms | Sumber: {final_source}")
                
            detection_buffer = []
            latencies = []
            start_timer = time.time()

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nSistem multi-model dihentikan.")

cap.release()