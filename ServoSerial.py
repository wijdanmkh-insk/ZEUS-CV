import cv2
import time
import torch
import serial
from collections import Counter
from PIL import Image
from torchvision import transforms

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
# TORCHSCRIPT MODEL (QUANTIZED)
# =========================
model = torch.jit.load("model.pth", map_location=device)
model.eval()

# =========================
# CLASS NAMES (HARUS SINKRON DENGAN TRAINING)
# =========================
class_names = [
    "Background",
    "Cardboard",
    "Food Organics",
    "Glass",
    "Hazard",
    "Metal",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]

NUM_CLASSES = len(class_names)

print("📦 Classes:", NUM_CLASSES)

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
# LABEL MAP
# =========================
def map_waste(label):
    if label in ["Vegetation", "Food Organics"]:
        return "ORGANIC"
    elif label in ["Paper", "Cardboard"]:
        return "PAPER"
    elif label in ["Glass", "Metal"]:
        return "HAZARD"
    elif label in ["Plastic", "Textile Trash"]:
        return "ANORGANIC"
    return "UNKNOWN"

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(1)

# =========================
# STATE
# =========================
votes = []
start_time = time.time()
last_send = 0

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(rgb)
    x = transform(img).unsqueeze(0).to(device)

    # =========================
    # INFERENCE (TORCHSCRIPT ONLY)
    # =========================
    with torch.no_grad():
        out = model(x)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)

    pred_idx = pred.item()

    # SAFETY CHECK
    if pred_idx >= NUM_CLASSES:
        print("⚠️ Invalid class index:", pred_idx)
        continue

    label = class_names[pred_idx]
    confidence = conf.item()
    waste = map_waste(label)

    # =========================
    # VOTING
    # =========================
    if confidence >= CONF_THRESHOLD:
        votes.append(waste)
    else:
        votes.append("UNKNOWN")

    # =========================
    # WINDOW DECISION
    # =========================
    if time.time() - start_time >= VOTE_WINDOW:

        if votes:
            count = Counter(votes)
            final, freq = count.most_common(1)[0]
            stability = freq / len(votes)

            if final != "UNKNOWN":
                if time.time() - last_send > SERVO_COOLDOWN:

                    print(f"\n🔒 FINAL: {final} | {stability*100:.1f}%")
                    send_command(final)

                    last_send = time.time()

        votes.clear()
        start_time = time.time()

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame,
                f"{label} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                f"Buffer: {len(votes)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    cv2.imshow("AI Waste System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

if arduino:
    arduino.close()