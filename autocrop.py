import time
import cv2
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CAMERA_INDEX = 0
YOLO_MODEL_PATH = 'yolov8n.pt'
CLASSIFIER_MODEL_PATH = 'V1_deploy.pt'

CLASS_NAMES = {
    0: "Background", 1: "Cardboard", 2: "Ewaste", 3: "Food Organics", 4: "Glass", 
    5: "Hazardous", 6: "Metal", 7: "Paper", 8: "Plastic", 9: "Textile Trash"
}

# Threshold deteksi gerakan OpenCV (makin kecil makin sensitif)
MOTION_THRESHOLD = 10000  

# ==============================================================================
# INITIALIZATION
# ==============================================================================
print("[ZEUS] Menginisialisasi sistem headless...")
device = torch.device('cpu')

# Muat model AI
yolo_model = YOLO(YOLO_MODEL_PATH)
try:
    classifier_model = torch.jit.load(CLASSIFIER_MODEL_PATH, map_location=device)
    print("[ZEUS] Model TorchScript MobileNet berhasil dimuat.")
except Exception:
    classifier_model = torch.load(CLASSIFIER_MODEL_PATH, map_location=device, weights_only=False)

classifier_model.eval()

# Preprocessing MobileNet
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

# Inisialisasi background untuk deteksi gerakan
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

        # 1. DETEKSI GERAKAN (Sangat Ringan, < 1ms CPU)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # Hitung perbedaan antar frame
        frame_delta = cv2.absdiff(gray_background, gray_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Hitung berapa banyak piksel yang berubah
        motion_pixel_count = cv2.countNonZero(thresh)

        # Update background secara perlahan agar adaptif terhadap perubahan cahaya lab
        cv2.addWeighted(gray_frame, 0.05, gray_background, 0.95, 0, gray_background)

        # 2. TRIGGER STATE (Jika ada pergerakan signifikan)
        if motion_pixel_count > MOTION_THRESHOLD:
            print("[TRIGGER] Pergerakan terdeteksi! Mengunci posisi sampah...")
            
            # Berhenti sejenak (0.5 detik) agar gerakan objek stabil/berhenti bergerak
            time.sleep(0.5)
            
            # Ambil frame terbaru yang jernih (tidak blur karena pergerakan)
            for _ in range(5):  # Flush buffer kamera agar dapat frame paling baru
                cap.read()
            ret, capture_frame = cap.read()
            if not ret:
                continue

            h_f, w_f, _ = capture_frame.shape

            # 3. RUN YOLO (Hanya 1x eksekusi)
            print("[AI EXEC] Menjalankan YOLOv8 Localizer...")
            t_start_yolo = time.time()
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

            # 4. RUN MOBILENET CLASSIFIER (Hanya 1x eksekusi jika objek ketemu)
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cropped_object = capture_frame[y1:y2, x1:x2]

                print("[AI EXEC] Mengklasifikasikan gambar crop dengan MobileNet...")
                t_start_clf = time.time()
                
                input_tensor = transform(cropped_object).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = classifier_model(input_tensor)
                
                t_end_ai = time.time()

                # Hitung Hasil
                _, predicted = torch.max(output, 1)
                final_class = CLASS_NAMES.get(predicted.item(), f"Unknown ({predicted.item()})")
                
                yolo_time = (t_start_clf - t_start_yolo) * 1000
                clf_time = (t_end_ai - t_start_clf) * 1000
                total_time = yolo_time + clf_time

                # OUTPUT UTAMA (Bisa disambungkan ke logika GPIO / Servo dari sini)
                print("-" * 50)
                print(f"HASIL DETEKSI : {final_class.upper()}")
                print(f"YOLO Latency  : {yolo_time:.1f} ms")
                print(f"MobileNet Lat : {clf_time:.1f} ms")
                print(f"Total AI Time : {total_time:.1f} ms")
                print("-" * 50)

                # --- TEMPATKAN LOGIKA KONTROL HARDWARE DI SINI ---
                # Contoh: jika final_class == "Plastic": putar_servo_ke_kanan()
                
                # Jeda Pengosongan (Cooldown 4 detik agar aktuator bekerja 
                # dan mencegah double trigger saat sampah jatuh)
                print("[SISTEM] Cooldown. Menunggu sistem siap kembali...")
                time.sleep(4.0)
                
                # Reset background pasca-eksekusi agar tidak mendeteksi sisa gerakan
                for _ in range(5): cap.read()
                ret, reset_frame = cap.read()
                if ret:
                    gray_background = cv2.cvtColor(reset_frame, cv2.COLOR_BGR2GRAY)
                    gray_background = cv2.GaussianBlur(gray_background, (21, 21), 0)
                
                print("\n=== SISTEM READY (IDLE MODE) ===")
            else:
                print("[INFO] Gerakan terdeteksi tapi YOLO tidak menemukan objek valid.\n")

        # Batasi loop idle agar tidak memakan spin-lock CPU
        time.sleep(0.03)

except KeyboardInterrupt:
    print("\n[ZEUS] Mematikan sistem headless.")
finally:
    cap.release()
    print("[ZEUS] Selesai.")