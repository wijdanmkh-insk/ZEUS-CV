import os
import cv2
import time
import torch
import serial
import signal
import sys
import torchvision.models as models
from collections import deque, Counter

from PIL import Image
from torchvision import transforms

# =========================
# HEADLESS EXIT HANDLER
# =========================
def handle_exit(sig, frame):
    print("\n🛑 Stopping gracefully...")
    cap.release()
    arduino.close()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESHOLD = 0.75
SERVO_COOLDOWN = 5.0

# =========================
# SERIAL TO ARDUINO
# =========================
arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)
print("✅ Arduino connected")

# =========================
# LOAD MODEL
# =========================
model_path = os.path.join(os.path.dirname(__file__), "best_zeus_model.pth")
checkpoint = torch.load(model_path, map_location=device)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, 10)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print("✅ Model loaded")

# =========================
# CLASS NAMES & TRANSFORM
# =========================
class_names = [
    "Cardboard", "Food Organics", "Glass", "Metal",
    "Miscellaneous Trash", "Paper", "Plastic",
    "Textile Trash", "Vegetation", "Background"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def map_waste_category(label):
    if label in ["Vegetation", "Food Organics"]:
        return "ORGANIC"
    elif label in ["Paper", "Cardboard"]:
        return "PAPER"
    elif label in ["Glass", "Metal"]:
        return "HAZARD"
    elif label in ["Miscellaneous Trash", "Plastic", "Textile Trash"]:
        return "ANORGANIC"
    return None

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Camera gagal dibuka")
    sys.exit(1)
print("📷 Camera connected")

# =========================
# STABILITY VARIABLES
# =========================
BUFFER_SIZE = 30
prediction_buffer = deque(maxlen=BUFFER_SIZE)
MAJORITY_THRESHOLD = 0.80

last_sent_time = 0
current_locked_waste = None

print("🚀 Running in headless mode. Press Ctrl+C to stop.\n")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame gagal dibaca")
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = conf.item()
    waste_type = map_waste_category(label)
    current_time = time.time()

    if confidence >= CONF_THRESHOLD and waste_type is not None:
        prediction_buffer.append(waste_type)
    else:
        prediction_buffer.append("UNKNOWN")

    # =========================
    # LOCKED & SMOOTHING LOGIC
    # =========================
    most_common_waste = None
    confidence_in_buffer = 0.0

    if len(prediction_buffer) == BUFFER_SIZE:
        counts = Counter(prediction_buffer)
        most_common_waste, occurrence = counts.most_common(1)[0]
        confidence_in_buffer = occurrence / BUFFER_SIZE

        if (
            most_common_waste != "UNKNOWN"
            and confidence_in_buffer >= MAJORITY_THRESHOLD
            and (current_time - last_sent_time) >= SERVO_COOLDOWN
        ):
            current_locked_waste = most_common_waste
            print(f"🔒 LOCKED & SENDING -> {current_locked_waste} "
                  f"(Consistency: {confidence_in_buffer*100:.1f}%)")
            arduino.write((current_locked_waste + "\n").encode())

            last_sent_time = current_time
            prediction_buffer.clear()

    # =========================
    # LOGGING (pengganti OSD)
    # =========================
    # Uncomment baris di bawah jika ingin verbose log setiap frame
    # print(f"Raw: {label} ({confidence:.2f}) | Buffer: {most_common_waste}")