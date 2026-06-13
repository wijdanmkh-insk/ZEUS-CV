import time
import torch
import cv2
import torchvision.transforms as transforms
from collections import Counter

# --- KONFIGURASI ---
CAMERA_INDEX = 0
MODEL_PATH = 'V1_deploy.pt'

class_names = {
    0: "Background", 1: "Cardboard", 2: "Ewaste", 3: "Food Organics", 4: "Glass", 
    5: "Hazardous", 6: "Metal", 7: "Paper", 8: "Plastic", 9: "Textile Trash"
}

# --- INSILISASI DEVICE & MODEL ---
device = torch.device('cpu')

try:
    # Menggunakan torch.jit.load karena V1_deploy.pt adalah TorchScript archive
    model = torch.jit.load(MODEL_PATH, map_location=device)
    print(f"Model TorchScript berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    print(f"Gagal memuat model dengan torch.jit.load. Mencoba fallback torch.load...")
    # Fallback jika ternyata membutuhkan weights_only=False
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model.eval()

print(f"Device yang digunakan: {device.type.upper()}")
print(f"Membuka Kamera Indeks: {CAMERA_INDEX}\n")

# --- PIPELINE PREPROCESSING ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- INISIALISASI KAMERA ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Kamera indeks {CAMERA_INDEX} tidak dapat diakses.")
    exit()

detection_buffer = []
start_timer = time.time()
final_display_text = "Mendeteksi..."
inference_time = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil gambar dari kamera.")
            break

        # Clone frame asli untuk visualisasi OpenCV agar tidak merusak preprocessing
        display_frame = frame.copy()

        # Preprocessing & Inference
        input_tensor = transform(frame).unsqueeze(0).to(device)

        t_start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        t_end = time.time()
        
        inference_time = (t_end - t_start) * 1000

        # Ambil Prediksi
        _, predicted = torch.max(output, 1)
        pred_class = predicted.item()
        detection_buffer.append(pred_class)

        # Logika Interval Evaluasi 3 Detik
        elapsed_time = time.time() - start_timer
        if elapsed_time >= 3.0:
            if detection_buffer:
                most_common_class = Counter(detection_buffer).most_common(1)[0][0]
                final_display_text = class_names.get(most_common_class, f"Class {most_common_class}")
                print(f"[HASIL 3s] Terbanyak: {final_display_text:<20} | Inference Terakhir: {inference_time:.2f} ms")
            detection_buffer = []
            start_timer = time.time()

        # --- RENDERING GUI OPENCV ---
        # Overlay Box Informasi (Semi-transparan background hitam)
        cv2.rectangle(display_frame, (10, 10), (420, 85), (0, 0, 0), -1)
        
        # Teks Hasil Klasifikasi & Waktu Inferensi
        cv2.putText(display_frame, f"Objek: {final_display_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"Inference: {inference_time:.1f} ms (CPU)", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Tampilkan Window
        cv2.imshow("ZEUS-CV Inference System", display_frame)

        # Batasan FPS GUI & Tombol Berhenti ('q' atau 'Esc')
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nProgram dihentikan melalui terminal.")

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Sistem GUI ditutup dengan bersih.")