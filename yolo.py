import cv2
import time
import torch
import serial
import numpy as np
from ultralytics import YOLO
import torchvision.models as models
from collections import Counter
from PIL import Image
from torchvision import transforms

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CONFIG
# =========================
CONF_THRESHOLD = 0.75
YOLO_CONF = 0.2

VOTE_WINDOW = 3.0
SERVO_COOLDOWN = 3.0

USE_SERIAL = True

# =========================
# SERIAL
# =========================
if USE_SERIAL:
    arduino = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
    time.sleep(2)
    print("✅ ESP32 connected")
else:
    arduino = None
    print("SIMULATION MODE")

def send_command(cmd):
    if USE_SERIAL and arduino:
        arduino.write((cmd + "\n").encode())
        arduino.flush()
    print("📡 SEND:", cmd)

# =========================
# YOLO MODEL
# =========================
yolo = YOLO("yolov8n.pt")  # ganti kalau sudah custom waste model

# =========================
# CLASSIFIER (MobileNet)
# =========================
checkpoint = torch.load("best_zeus_model.pth", map_location=device)
class_names = checkpoint["class_names"]

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# WASTE MAP
# =========================
def map_waste(label):
    if label in ["Vegetation", "Food Organics"]:
        return "ORGANIC"
    elif label in ["Paper", "Cardboard"]:
        return "PAPER"
    elif label in ["Glass", "Metal"]:
        return "HAZARD"
    elif label in ["Plastic", "Textile Trash", "Miscellaneous Trash"]:
        return "ANORGANIC"
    return None

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

# =========================
# VOTING STATE
# =========================
votes = []
start_time = time.time()
last_send_time = 0

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # =========================
    # YOLO DETECTION
    # =========================
    results = yolo(frame, verbose=False)[0]
    boxes = results.boxes

    waste = None
    confidence = 0

    if boxes is not None and len(boxes) > 0:

        # ambil box terbaik
        best_box = max(boxes, key=lambda b: float(b.conf[0]))

        conf_yolo = float(best_box.conf[0])

        if conf_yolo > YOLO_CONF:

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size != 0:

                # =========================
                # CLASSIFICATION
                # =========================
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                x = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(x)
                    prob = torch.nn.functional.softmax(out, dim=1)
                    conf, pred = torch.max(prob, 1)

                label = class_names[pred.item()]
                confidence = conf.item()
                waste = map_waste(label)

                # draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

    # =========================
    # VOTING INPUT
    # =========================
    if confidence >= CONF_THRESHOLD and waste:
        votes.append(waste)
    else:
        votes.append("UNKNOWN")

    # =========================
    # VOTING WINDOW (3 detik)
    # =========================
    if time.time() - start_time >= VOTE_WINDOW:

        if len(votes) > 0:
            count = Counter(votes)
            final, freq = count.most_common(1)[0]
            stability = freq / len(votes)

            if final != "UNKNOWN":
                if time.time() - last_send_time > SERVO_COOLDOWN:

                    print(f"\n🔒 FINAL: {final} | {stability*100:.1f}%")
                    send_command(final)

                    last_send_time = time.time()

        votes.clear()
        start_time = time.time()

    # =========================
    # UI
    # =========================
    cv2.putText(frame,
                f"Detect: {waste} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    cv2.putText(frame,
                f"Buffer: {len(votes)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)

    cv2.imshow("YOLO + Waste Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

if USE_SERIAL and arduino:
    arduino.close()