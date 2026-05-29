import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# =========================
# 1. CONFIG
# =========================

#ip_camera_url = "https://192.168.1.21:8080/video"

# atau kalau HTTP:
# ip_camera_url = "http://192.168.1.10:81/stream"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. LOAD MODEL (.pth)
# =========================


model_path = os.path.join(os.path.dirname(__file__), "classify.pth")
checkpoint = torch.load(model_path, map_location=device)

# NOTE: Anda harus menginisialisasi arsitektur model sebelum me-load state_dict.
# Dari error traceback, arsitektur yang digunakan adalah MobileNetV2, bukan ResNet18
import torchvision.models as models
model = models.mobilenet_v2(weights=None)
# Sesuaikan jumlah output features dengan jumlah class yang ada di checkpoint (9)
model.classifier[1] = torch.nn.Linear(model.last_channel, 9)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("✅ Model loaded")

# class names HARUS sama seperti training kamu
class_names = ["Cardboard", "Food Organics", "Glass", "Metal", "Miscellaneous Trash", "Paper", "Plastic", "Textile Trash", "Vegetation"]

# =========================
# 3. TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# 4. OPEN IP CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Tidak bisa buka IP Camera")
    exit()

print("📷 IP Camera connected")

# =========================
# 5. REALTIME LOOP
# =========================

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame tidak terbaca")
        break

    # resize biar ringan
    frame = cv2.resize(frame, (640, 480))

    # BGR → RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # preprocess
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = conf.item()

    # =========================
    # 6. DRAW RESULT
    # =========================

    text = f"{label} ({confidence:.2f})"

    cv2.putText(frame,
                text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("IP Camera Classification", frame)

    # tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# 7. CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()