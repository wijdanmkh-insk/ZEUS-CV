import os
import cv2
import time
import torch
import serial
import torchvision.models as models

from PIL import Image
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESHOLD = 0.80
STABLE_TIME = 4.0
SERVO_COOLDOWN = 5.0

# =========================
# SERIAL TO ARDUINO
# =========================

# WINDOWS:
arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# LINUX / RASPBERRY PI:
# arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

time.sleep(2)

print("✅ Arduino connected")

# =========================
# LOAD MODEL
# =========================

model_path = os.path.join(os.path.dirname(__file__), "best_zeus_model.pth")

checkpoint = torch.load(
    model_path,
    map_location=device
)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(
    model.last_channel,
    10
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("✅ Model loaded")
# =========================
# CLASS NAMES
# =========================

class_names = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
    "Background"
]

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


def map_waste_category(label):

    if label in ["Vegetation", "Food Organics"]:
        return "ORGANIC"

    elif label == "Paper":
        return "PAPER"

    elif label in ["Glass", "Metal"]:
        return "HAZARD"

    elif label in [
        "Cardboard",
        "Miscellaneous Trash",
        "Plastic",
        "Textile Trash"
    ]:
        return "ANORGANIC"

    return None

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Camera gagal dibuka")
    exit()

print("📷 Camera connected")

# =========================
# STABILITY VARIABLES
# =========================

stable_class = None
stable_start_time = None
last_sent_time = 0

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Frame gagal dibaca")
        break

    frame = cv2.resize(frame, (640, 480))

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(rgb)

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probs = torch.nn.functional.softmax(
            outputs,
            dim=1
        )

        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = conf.item()

    waste_type = map_waste_category(label)

    current_time = time.time()

    # =========================
    # STABILITY CHECK
    # =========================

    if (
        confidence >= CONF_THRESHOLD
        and waste_type is not None
    ):

        if waste_type == stable_class:

            elapsed = current_time - stable_start_time

            if (
                elapsed >= STABLE_TIME
                and
                (current_time - last_sent_time)
                >= SERVO_COOLDOWN
            ):

                print(
                    f"Sending -> {waste_type}"
                )

                arduino.write(
                    (waste_type + "\n").encode()
                )

                last_sent_time = current_time

                stable_class = None
                stable_start_time = None

        else:

            stable_class = waste_type
            stable_start_time = current_time

    else:

        stable_class = None
        stable_start_time = None

    # =========================
    # DRAW TEXT
    # =========================

    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Category: {waste_type}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    if (
        stable_class is not None
        and
        stable_start_time is not None
    ):

        remaining = (
            STABLE_TIME
            -
            (current_time - stable_start_time)
        )

        if remaining < 0:
            remaining = 0

        cv2.putText(
            frame,
            f"Locking: {stable_class} ({remaining:.1f}s)",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    cv2.imshow(
        "Waste Classification",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()

arduino.close()