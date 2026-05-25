import cv2
import serial
import time
from ultralytics import YOLO

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

MODEL_PATH = 'best.pt'

CLASS_MAPPING = {
    'organic': 'O',
    'anorganic': 'W',
    'paper': 'P',
    'hazard': 'H'
}

print("Connecting to Arduino...")
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

model = YOLO(MODEL_PATH, task='detect')
cap = cv2.VideoCapture(1)

last_sent = 0
COOLDOWN = 4.0

stable_label = None
stable_count = 0

print("System running...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.2, device='cpu')

    detected = None

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id].lower()

            if name in CLASS_MAPPING:
                detected = CLASS_MAPPING[name]

    # stability filter (important fix)
    if detected == stable_label:
        stable_count += 1
    else:
        stable_label = detected
        stable_count = 0

    now = time.time()

    if stable_label and stable_count > 5:
        if now - last_sent > COOLDOWN:
            msg = stable_label + "\n"
            arduino.write(msg.encode())
            print("Sent:", stable_label)
            last_sent = now
            stable_count = 0

    cv2.imshow("YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()