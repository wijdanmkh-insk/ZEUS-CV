import os
import shutil
from ultralytics import YOLO

# 1. Muat model .pt hasil training kamu (Awalnya masuk ke GPU jika ada)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Paksa pakai GPU 0, tapi nanti kita pindahin ke CPU
model = YOLO("ref/best.pt")

# 2. JINAKKAN KE CPU DULU (Wajib sebelum diekspor!)
model.to("cpu")

# 3. Jalankan ekspor (Sekarang ONNX akan di-generate murni dengan binding CPU)
exported_path = model.export(format="onnx", imgsz=320, half=True, simplify=True)

# 4. Trik pindahin manual ke folder 'model' biar gak acak-acakan bawaan YOLO
os.makedirs("model", exist_ok=True)
shutil.move(exported_path, "model/best.onnx")

print("Dah kelar, Dan! Cek folder 'model/best.onnx' murni versi CPU.")