import cv2
import time
from ultralytics import YOLO

# Load model yang sudah kamu kuantisasi (ONNX / OpenVINO)
model = YOLO("../model/best.onnx", task="detect")

cap = cv2.VideoCapture(0)

print("--- MEMULAI PENGUJIAN LATENSI ZEUS ---")
print("Format data: Latency (ms) | FPS")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # SINKRONISASI & HITUNG WAKTU MULAI (Pre-Inference)
    # Gunakan time.perf_counter() karena jauh lebih presisi untuk benchmark hardware
    start_inference = time.perf_counter()
    
    # PROSES INFERENSI AI (Bagian yang diuji untuk kalibrasi)
    results = model(frame, device="cpu", imgsz=320, verbose=False)
    
    # HITUNG WAKTU SELESAI (Post-Inference)
    end_inference = time.perf_counter()
    
    # Hitung durasi murni inferensi dalam satuan milidetik (ms)
    # Dikali 1000 untuk mengubah detik ke milidetik
    inference_latency_ms = (end_inference - start_inference) * 1000
    
    # Hitung FPS teoretis yang didasarkan HANYA pada kecepatan model berpikir
    inference_fps = 1000 / inference_latency_ms if inference_latency_ms > 0 else 0

    # Cetak ke terminal agar mudah di-copy-paste ke Excel/Spreadsheet untuk tabel data
    print(f"{inference_latency_ms:.2f} ms | {inference_fps:.1f} FPS")
    
    # Tampilkan bounding box jika ada objek sampah terdeteksi
    if results and len(results) > 0:
        frame = results[0].plot()
        
    # Tampilkan visualisasi latensi di layar kamera (Warna kuning taktis)
    latency_text = f"Latency AI: {inference_latency_ms:.1f} ms"
    fps_text = f"Inference FPS: {inference_fps:.1f}"
    
    cv2.putText(frame, latency_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
    cv2.imshow("ZEUS Calibration Room", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()