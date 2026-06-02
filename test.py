import cv2
import time
import torch
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
SERVO_COOLDOWN = 5.0

BUFFER_SIZE = 30
MAJORITY_THRESHOLD = 0.80

# =========================
# LOAD MODEL
# =========================
checkpoint = torch.load("best_zeus_model.pth", map_location=device)

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
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# WASTE CATEGORY MAPPING
# =========================
def map_waste_category(label):
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
# SIMULATED SERIAL SEND
# =========================
def send_to_arduino_sim(data):
    print(f"📡 [SIMULATED SERIAL] -> {data}")

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera gagal dibuka")
    exit()

print("📷 Camera ready")

# =========================
# BUFFER
# =========================
prediction_buffer = deque(maxlen=BUFFER_SIZE)

last_sent_time = 0

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
    input_tensor = transform(img).unsqueeze(0).to(device)

    # =========================
    # PREDICT
    # =========================
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = conf.item()

    waste_type = map_waste_category(label)
    current_time = time.time()

    # =========================
    # BUFFERING
    # =========================
    if confidence >= CONF_THRESHOLD and waste_type is not None:
        prediction_buffer.append(waste_type)
    else:
        prediction_buffer.append("UNKNOWN")

    # =========================
    # DECISION LOGIC
    # =========================
    most_common = None
    stability = 0

    if len(prediction_buffer) == BUFFER_SIZE:
        counts = Counter(prediction_buffer)
        most_common, count = counts.most_common(1)[0]
        stability = count / BUFFER_SIZE

        if most_common != "UNKNOWN" and stability >= MAJORITY_THRESHOLD:
            print("\n==============================")
            print(f"🔒 LOCKED RESULT: {most_common}")
            print(f"📊 Stability: {stability*100:.1f}%")
            print(f"🎯 Raw: {label} ({confidence:.2f})")
            print("==============================\n")

            # SIMULASI SERIAL OUTPUT
            send_to_arduino_sim(most_common)

            prediction_buffer.clear()

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame,
                f"{label} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0), 2)

    if most_common:
        cv2.putText(frame,
                    f"Stable: {most_common} ({stability*100:.1f}%)",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0), 2)

    cv2.imshow("SIMULATION MODE", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()