import cv2
import time
import torch
import serial
import torchvision.models as models
from collections import deque, Counter
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
BUFFER_SIZE = 20                 # smoothing window
MAJORITY_THRESHOLD = 0.70       # lebih fleksibel untuk "ragu"

SERVO_DELAY = 3.0              # ⬅️ delay sebelum eksekusi servo
SERVO_COOLDOWN = 5.0

# =========================
# SERIAL (GANTI SIMULASI JIKA BELUM ADA ARDUINO)
# =========================
USE_SERIAL = False  # ⬅️ ubah True kalau sudah pakai Arduino

if USE_SERIAL:
    port = "/dev/ttyUSB0"
    arduino = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)
    print("✅ Arduino connected")
else:
    print("🧪 RUNNING IN SIMULATION MODE")

def send_command(cmd):
    if USE_SERIAL:
        arduino.write((cmd + "\n").encode())
    print(f"📡 SEND -> {cmd}")

# =========================
# LOAD MODEL
# =========================
checkpoint = torch.load("last-test.pth", map_location=device)

class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("✅ Model loaded")

# =========================
# TRANSFORM
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
# MAPPING
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
print("📷 Camera ready")

# =========================
# BUFFER SYSTEM (SMOOTHING)
# =========================
buffer = deque(maxlen=BUFFER_SIZE)

# =========================
# STATE CONTROL
# =========================
locked_class = None
lock_time = 0
last_servo_time = 0

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(rgb)
    x = transform(img).unsqueeze(0).to(device)

    # =========================
    # INFERENCE
    # =========================
    with torch.no_grad():
        out = model(x)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)

    label = class_names[pred.item()]
    confidence = conf.item()
    waste = map_waste(label)

    # =========================
    # BUFFER UPDATE
    # =========================
    if confidence >= CONF_THRESHOLD and waste:
        buffer.append(waste)
    else:
        buffer.append("UNKNOWN")

    # =========================
    # DECISION (RATA-RATA / VOTING)
    # =========================
    final = None
    stability = 0

    if len(buffer) == BUFFER_SIZE:
        count = Counter(buffer)
        final, c = count.most_common(1)[0]
        stability = c / BUFFER_SIZE

        # =========================
        # LOCK DETECTION
        # =========================
        if (
            final != "UNKNOWN"
            and stability >= MAJORITY_THRESHOLD
            and (time.time() - last_servo_time) > SERVO_COOLDOWN
        ):
            locked_class = final
            lock_time = time.time()
            print(f"\n🔒 LOCKED: {locked_class} ({stability*100:.1f}%)")

            buffer.clear()

    # =========================
    # DELAY EXECUTION (3 SECONDS)
    # =========================
    if locked_class:
        elapsed = time.time() - lock_time

        if elapsed >= SERVO_DELAY:
            print(f"🚀 EXECUTE SERVO -> {locked_class}")
            send_command(locked_class)

            last_servo_time = time.time()
            locked_class = None

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame, f"Raw: {label} ({confidence:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if final:
        cv2.putText(frame,
                    f"Stable: {final} ({stability*100:.1f}%)",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255,255,0), 2)

    if locked_class:
        cv2.putText(frame,
                    f"LOCKED: {locked_class} ({int(SERVO_DELAY - (time.time() - lock_time))}s)",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,0,255), 2)

    cv2.imshow("AI Waste Sorting System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
if USE_SERIAL:
    arduino.close()