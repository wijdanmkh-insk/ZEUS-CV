import os
import cv2
import time
import torch
import serial
import torchvision.models as models
from collections import deque, Counter  # <-- Tambahkan Counter dan deque

from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESHOLD = 0.75  # Sedikit diturunkan agar tidak terlalu ketat karena sudah dibantu buffer
SERVO_COOLDOWN = 5.0

# =========================
# SERIAL TO ARDUINO
# =========================
# Sesuaikan port dengan OS yang aktif
port = "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
arduino = serial.Serial(f'/dev/{port}', 115200, timeout=1)
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
# CLASS NAMES
# =========================
class_names = [
    "Cardboard", "Food Organics", "Glass", "Metal", 
    "Miscellaneous Trash", "Paper", "Plastic", 
    "Textile Trash", "Vegetation", "Background"
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
    exit()
print("📷 Camera connected")

# =========================
# STABILITY VARIABLES (BUFFER APPROACH)
# =========================
# Menggunakan deque untuk menyimpan riwayat prediksi 2 detik terakhir.
# Jika webcam berjalan di ~15-20 FPS, 30 frame setara dengan sekitar 1.5 - 2 detik.
BUFFER_SIZE = 30 
prediction_buffer = deque(maxlen=BUFFER_SIZE)

# Persentase kemunculan minimum dalam buffer untuk dianggap "Locked" (80% dari BUFFER_SIZE)
MAJORITY_THRESHOLD = 0.80 

last_sent_time = 0
current_locked_waste = None

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

    # Masukkan ke buffer jika confidence masuk akal dan bukan background/None
    if confidence >= CONF_THRESHOLD and waste_type is not None:
        prediction_buffer.append(waste_type)
    else:
        # Opsional: masukkan "Unknown" agar riwayat lama tergeser jika objek hilang
        prediction_buffer.append("UNKNOWN")

    # =========================
    # LOCKED & SMOOTHING LOGIC
    # =========================
    most_common_waste = None
    confidence_in_buffer = 0.0

    if len(prediction_buffer) == BUFFER_SIZE:
        # Hitung kemunculan tiap kelas di dalam buffer
        counts = Counter(prediction_buffer)
        most_common_waste, occurrence = counts.most_common(1)[0]
        
        # Hitung rasio konsistensi kelas dominan tersebut
        confidence_in_buffer = occurrence / BUFFER_SIZE

        # Jika lolos batas mayoritas dan servo sedang tidak cooldown
        if (
            most_common_waste != "UNKNOWN" 
            and confidence_in_buffer >= MAJORITY_THRESHOLD
            and (current_time - last_sent_time) >= SERVO_COOLDOWN
        ):
            current_locked_waste = most_common_waste
            
            print(f"🔒 LOCKED & SENDING -> {current_locked_waste} (Consistency: {confidence_in_buffer*100:.1f}%)")
            arduino.write((current_locked_waste + "\n").encode())
            
            last_sent_time = current_time
            # Flush buffer setelah eksekusi agar tidak terjadi double-trigger berturut-turut
            prediction_buffer.clear() 

    # =========================
    # DRAW TEXT & OSD
    # =========================
    # Tampilkan pembacaan instan (raw)
    cv2.putText(frame, f"Raw: {label} ({confidence:.2f})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tampilkan isi buffer/status penguncian
    if most_common_waste and most_common_waste != "UNKNOWN":
        cv2.putText(frame, f"Buffer Dominant: {most_common_waste} ({confidence_in_buffer*100:.1f}%)", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Buffer: Stabilizing...", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Indikator Cooldown Servo
    cooldown_rem = SERVO_COOLDOWN - (current_time - last_sent_time)
    if cooldown_rem > 0:
        cv2.putText(frame, f"Servo Cooldown: {cooldown_rem:.1f}s", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(frame, "Servo: READY", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 254, 0), 2)

    cv2.imshow("Waste Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
arduino.close()