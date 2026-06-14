import time
import cv2
import torch
import torchvision.transforms as transforms
import serial  
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CAMERA_INDEX = 1
YOLO_MODEL_PATH = 'yolov8n.pt'
CLASSIFIER_MODEL_PATH = 'V1_deploy.pt'

SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200

CLASS_NAMES = {
    0: "Background", 1: "Cardboard", 2: "Ewaste", 3: "Food Organics", 4: "Glass", 
    5: "Hazardous", 6: "Metal", 7: "Paper", 8: "Plastic", 9: "Textile Trash"
}

# --- PEMETAAN KLUSTER SAMPAH (Key dipastikan sesuai dengan teks di CLASS_NAMES) ---
CLUSTER_MAPPING = {
    "background": "UNKNOWN",
    "cardboard": "PAPER",
    "paper": "PAPER",
    "plastic": "ANORGANIC",
    "textile trash": "ANORGANIC",
    "hazardous": "HAZARD",
    "glass": "HAZARD",
    "ewaste": "HAZARD",
    "metal": "HAZARD",
    "food organics": "ORGANIC"
}

MOTION_THRESHOLD = 10000  

# ==============================================================================
# INITIALIZATION
# ==============================================================================
print("[ZEUS] Menginisialisasi sistem headless dengan Serial...")
device = torch.device('cpu')

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  
    print(f"[SERIAL] Berhasil terhubung ke ESP32 via {SERIAL_PORT}")
except Exception as e:
    print(f"[SERIAL ERROR] Gagal membuka port {SERIAL_PORT}: {e}")
    print("[SERIAL WARNING] Program akan tetap berjalan tanpa mengirim data fisik.")
    ser = None

yolo_model = YOLO(YOLO_MODEL_PATH)
try:
    classifier_model = torch.jit.load(CLASSIFIER_MODEL_PATH, map_location=device)
    print("[ZEUS] Model TorchScript MobileNet berhasil dimuat.")
except Exception:
    classifier_model = torch.load(CLASSIFIER_MODEL_PATH, map_location=device, weights_only=False)

classifier_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"[ERROR] Kamera indeks {CAMERA_INDEX} tidak dapat diakses.")
    exit()

ret, first_frame = cap.read()
if ret:
    gray_background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.GaussianBlur(gray_background, (21, 21), 0)
else:
    print("[ERROR] Gagal mengambil frame awal kamera.")
    exit()

print("\n=== SISTEM ZEUS READY (HEADLESS IDLE MODE) ===")
print("Menunggu objek dijatuhkan ke dalam tempat sampah...\n")

# ==============================================================================
# MAIN LOOP (STATE MACHINE)
# ==============================================================================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        frame_delta = cv2.absdiff(gray_background, gray_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        motion_pixel_count = cv2.countNonZero(thresh)
        cv2.addWeighted(gray_frame, 0.05, gray_background, 0.95, 0, gray_background)

        if motion_pixel_count > MOTION_THRESHOLD:
            print("[TRIGGER] Pergerakan terdeteksi! Menunggu objek stabil (2 detik)...")
            time.sleep(2.0)
            
            print("[SISTEM] Mengambil frame terbaru...")
            for _ in range(10):  
                cap.read()
                
            ret, capture_frame = cap.read()
            if not ret:
                continue

            h_f, w_f, _ = capture_frame.shape

            print("[AI EXEC] Menjalankan YOLOv8 Localizer...")
            yolo_results = yolo_model(capture_frame, verbose=False)[0]
            
            best_box = None
            max_area = 0

            for box in yolo_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_f, x2), min(h_f, y2)
                
                area = (x2 - x1) * (y2 - y1)
                if area > max_area and area > 1500:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cropped_object = capture_frame[y1:y2, x1:x2]

                print("[AI EXEC] Mengklasifikasikan gambar dengan MobileNet...")
                input_tensor = transform(cropped_object).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = classifier_model(input_tensor)
                
                _, predicted = torch.max(output, 1)
                detected_object_name = CLASS_NAMES.get(predicted.item(), "Background")
                
                # --- PERBAIKAN: Paksa cari dengan huruf kecil agar selalu cocok ---
                search_key = detected_object_name.lower().strip()
                target_cluster = CLUSTER_MAPPING.get(search_key, "UNKNOWN")

                print("-" * 50)
                print(f"OBJEK TERDETEKSI : {detected_object_name.upper()}")
                print(f"KLUSTER SAMPAH   : {target_cluster}")
                print("-" * 50)

                # --- KIRIM PERINTAH SERIAL KE ESP32 ---
                if ser is not None and target_cluster != "UNKNOWN":
                    command_string = f"{target_cluster}\n"
                    ser.write(command_string.encode('utf-8'))
                    print(f"[SERIAL] -> TERKIRIM KE ESP32: {command_string.strip()}")

                    # --- COOLDOWN SYSTEM (Masuk ke dalam kondisi jika serial terkirim) ---
                    print(f"[SISTEM] Kamera dinonaktifkan sementara selama 5 detik untuk pergerakan mekanik...")
                    time.sleep(5.0)
                else:
                    print("[SERIAL] Perintah tidak dikirim karena kluster UNKNOWN atau port terputus.")
                
                print("[SISTEM] Mengkalibrasi ulang latar belakang...")
                for _ in range(15): 
                    cap.read()
                
                ret, reset_frame = cap.read()
                if ret:
                    gray_background = cv2.cvtColor(reset_frame, cv2.COLOR_BGR2GRAY)
                    gray_background = cv2.GaussianBlur(gray_background, (21, 21), 0)
                
                print("\n=== SISTEM READY (IDLE MODE) ===")
            else:
                print("[INFO] Gerakan terdeteksi tapi YOLO tidak menemukan objek valid.\n")

        time.sleep(0.03)

except KeyboardInterrupt:
    print("\n[ZEUS] Mematikan sistem headless.")
finally:
    cap.release()
    if ser is not None:
        ser.close()
        print("[SERIAL] Koneksi dihentikan.")
    print("[ZEUS] Selesai.")