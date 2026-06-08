import cv2
import time
import torch
import serial
from collections import Counter
from PIL import Image
from torchvision import transforms
import torchvision.models as models

# =========================
# DEVICE
# =========================
device = torch.device("cpu")

# =========================
# CONFIG
# =========================
CONF_THRESHOLD = 0.75
VOTE_WINDOW = 4.0
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

def send_command(cmd):
    if arduino:
        arduino.write((cmd + "\n").encode())
    print("📡 SEND:", cmd)

# =========================
# LOAD .pth CHECKPOINT
# =========================
ckpt = torch.load("model.pth", map_location=device)

state_dict = ckpt["model_state_dict"]
class_names = ckpt["class_names"]

NUM_CLASSES = len(class_names)
print("📦 Classes:", NUM_CLASSES)

# =========================
# MODEL
# =========================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)

model.load_state_dict(state_dict, strict=True)
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
# CAMERA (HEADLESS)
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not found")

# =========================
# STATE
# =========================
votes = []
start_time = time.time()
last_send = 0

print("🚀 HEADLESS INFERENCE RUNNING... (no GUI)")

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    x = transform(img).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)

    idx = pred.item()

    if idx >= NUM_CLASSES:
        continue

    label = class_names[idx]
    confidence = conf.item()

    # voting input
    if confidence >= CONF_THRESHOLD:
        votes.append(label)
    else:
        votes.append("UNKNOWN")

    # decision window
    if time.time() - start_time >= VOTE_WINDOW:

        if votes:
            final, freq = Counter(votes).most_common(1)[0]
            stability = freq / len(votes)

            if final != "UNKNOWN":
                if time.time() - last_send > SERVO_COOLDOWN:
                    print(f"\n🔒 FINAL: {final} | {stability*100:.1f}%")
                    send_command(final)
                    last_send = time.time()

            else:
                print("⚠️ No valid detection in window")

        votes.clear()
        start_time = time.time()

    time.sleep(0.01)

# =========================
# CLEANUP
# =========================
cap.release()
if arduino:
    arduino.close()