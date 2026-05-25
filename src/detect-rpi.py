import cv2
from ultralytics import YOLO
import typer
import time
import os
import json
import sys
from pathlib import Path

# ==============================================================================
# 🛠️ AREA DEBUGGING IMPORT SERIAL (Biar ketahuan kalau ada silent error)
# ==============================================================================
ZeusSerial = None
import_error_msg = ""

try:
    from src.serial_bridge import ZeusSerial
    print("✅ Successfully imported ZeusSerial from src.serial_bridge")
except Exception as e1:
    try:
        from serial_bridge import ZeusSerial
        print("✅ Successfully imported ZeusSerial from serial_bridge")
    except Exception as e2:
        import_error_msg = f"\n   - Hubungan 'src.serial_bridge': {str(e1)}\n   - Hubungan 'serial_bridge': {str(e2)}"
        ZeusSerial = None

# File configuration
JSON = os.path.join(os.path.dirname(__file__), "../res/detected.json")
CSV_LOG = os.path.join(os.path.dirname(__file__), "../res/latency_log.csv")

Path(os.path.dirname(JSON)).mkdir(parents=True, exist_ok=True)
REQ_TIME = 2.0

# Inisialisasi Model
model = YOLO("./model/rpi.onnx")  
class_names = model.names
print("✅ Model loaded: model/rpi.onnx")
print(f"📋 Class names: {class_names}")
print(f"📊 Total classes: {len(class_names)}")

# Tulis header CSV jika file belum ada
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w") as f:
        f.write("timestamp,latency_ms,inference_fps\n")

app = typer.Typer(help="YOLO detection on webcam using pure OpenCV rendering")

def save_detections(class_name: str):
    from datetime import datetime
    detections = []
    if os.path.exists(JSON):
        try:
            with open(JSON, 'r') as f:
                detections = json.load(f)
        except Exception:
            detections = []
    
    detections.append({
        "timestamp": datetime.now().isoformat(),
        "class": class_name
    })
    
    try:
        with open(JSON, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f"[SAVED] Deteksi '{class_name}' ke {JSON}")
    except Exception as e:
        print(f"⚠️ Error saving detections: {e}")

def process(source: str, conf_threshold: float = 0.6, required_time: float = REQ_TIME, serial_enable: bool = False, serial_port: str = '/dev/ttyUSB0', baudrate: int = 115200, mapping_path: str | None = None, debug: bool = False):
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    
    object_timers = {}
    logged_objects = set()

    serial_conn = None
    class_map = None
    
    if mapping_path:
        try:
            with open(mapping_path, 'r') as mf:
                class_map = json.load(mf)
                print(f"Loaded serial class mapping from {mapping_path}")
        except Exception as e:
            print(f"⚠️ Failed to load mapping file: {e}")

    # ==============================================================================
    # 🛠️ AREA HARD-DEBUGGING KONEKSI SERIAL (Analisis Penyebab Kegagalan)
    # ==============================================================================
    print("\n" + "="*50)
    print(f"[SERIAL DEBUG] Menjalankan inisialisasi serial...")
    print(f"[SERIAL DEBUG] Flag --serial aktif? -> {serial_enable}")
    print(f"[SERIAL DEBUG] Status Class ZeusSerial -> {'TERSEDIA' if ZeusSerial is not None else 'KOSONG/NULL'}")
    
    if serial_enable:
        if ZeusSerial is None:
            print("❌ KONEKSI BATAL: Modul serial_bridge gagal di-import total!")
            print(f"   Detail error saat import tadi:{import_error_msg}")
            print("   💡 SOLUSI: Pastikan lib 'pyserial' terinstal (`pip install pyserial`) atau file serial_bridge.py tidak error.")
        else:
            # Cek apakah port ada di sistem Linux RPi sebelum mencoba buka
            if not os.path.exists(serial_port):
                print(f"❌ KONEKSI BATAL: Port '{serial_port}' tidak ditemukan di sistem!")
                print("   💡 SOLUSI: Coba cabut-colok MCU, lalu ketik `ls /dev/tty*` di terminal RPi.")
                print("             Kemungkinan portnya berubah jadi `/dev/ttyACM0` atau sejenisnya.")
            else:
                try:
                    print(f"[SERIAL DEBUG] Mencoba membuka port {serial_port} dengan baudrate {baudrate}...")
                    serial_conn = ZeusSerial(port=serial_port, baudrate=baudrate)
                    print(f"✅ KONEKSI BERHASIL: Tersambung ke MCU via {serial_port}")
                except Exception as e:
                    print(f"❌ KONEKSI GAGAL: Terjadi masalah internal saat membuka port {serial_port}!")
                    print(f"   Detail Error Sistem: {str(e)}")
                    
                    # Analisis error berbasis teks bawaan OS Linux
                    if "Permission denied" in str(e) or "PermissionError" in str(e):
                        print("   💡 PENYEBAB: Hak akses diblokir oleh OS (Permission Denied).")
                        print(f"   💡 SOLUSI: Jalankan perintah ini di terminal RPi: `sudo chmod 666 {serial_port}`")
                    elif "Device or resource busy" in str(e):
                        print("   💡 PENYEBAB: Port sedang dipakai/dikunci oleh proses atau script lain!")
                        print(f"   💡 SOLUSI: Ketik `sudo lsof | grep {os.path.basename(serial_port)}` untuk cari PID-nya lalu bunuh prosesnya.")
                    
                    import traceback
                    print("\n--- Stack Trace Error Lengkap ---")
                    traceback.print_exc()
                    print("---------------------------------\n")
                    serial_conn = None
    print("="*50 + "\n")

    if not cap.isOpened():
        print(f"Error: Could not open source '{source}'.")
        return

    # Set resolusi kamera default
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_area = 640 * 480 

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to grab frame.")
                break

            annotated_frame = frame.copy()
            
            start_time = time.perf_counter() # Mulai hitung latensi
            
            results_list = model(frame, conf=conf_threshold, imgsz=640, device="cpu", verbose=False)
            results = results_list[0]
            
            end_time = time.perf_counter() # Selesai hitung latensi
            
            # Logika Latensi Riset
            latency_ms = (end_time - start_time) * 1000
            inf_fps = 1000 / latency_ms if latency_ms > 0 else 0
            
            with open(CSV_LOG, "a") as f:
                f.write(f"{time.time()},{latency_ms:.2f},{inf_fps:.1f}\n")

            boxes = results.boxes.xyxy.cpu().numpy()  
            scores = results.boxes.conf.cpu().numpy() 
            clss = results.boxes.cls.cpu().numpy()   
            
            if debug and len(boxes) > 0:
                print(f"[DEBUG] Detections found: {len(boxes)} | Latency: {latency_ms:.1f}ms")

            current_frame_classes = set()

            for box, score, cls in zip(boxes, scores, clss):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                
                class_name = class_names.get(class_id, str(class_id)) if isinstance(class_names, dict) else class_names[class_id]

                # Filter Kotak Raksasa
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                if box_area > (0.70 * frame_area):
                    continue  

                current_frame_classes.add(class_name)

                # Gambar Kotak Manual
                color = (0, 255, 0) 
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {score:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + len(label)*10, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Logika Timer dan Trigger Serial
                if class_name in logged_objects:
                    cv2.putText(annotated_frame, f"LOCKED: {class_name}", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    continue

                if class_name not in object_timers:
                    object_timers[class_name] = time.time()
                    elapsed_time = 0.0
                else:
                    elapsed_time = time.time() - object_timers[class_name]
                    
                cv2.putText(annotated_frame, f"Hold: {elapsed_time:.1f}s / {required_time}s", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                if elapsed_time >= required_time:
                    save_detections(class_name)
                    logged_objects.add(class_name)
                    
                    send_category = None
                    if class_map and class_name in class_map:
                        send_category = class_map[class_name]
                    else:
                        normalized = class_name.lower().replace(' ', '_')
                        if normalized in ("organic", "anorganic", "hazard", "paper", "anorganic_wet", "anorganic_dry"):
                            send_category = normalized
                        elif normalized == "o": send_category = "organic"
                        elif normalized == "p": send_category = "paper"

                    if serial_conn and send_category:
                        try:
                            serial_conn.send_trigger(send_category)
                        except Exception as e_send:
                            print(f"⚠️ Gagal mengirim data serial ke MCU: {e_send}")

            # Reset Timer untuk Objek yang Hilang
            for active_class in list(object_timers.keys()):
                if active_class not in current_frame_classes:
                    del object_timers[active_class]

            # Info Latensi di Layar Video
            cv2.putText(annotated_frame, f"Latency: {latency_ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Model FPS: {inf_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("ZEUS Live Cam - YOLO Manual Render", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        if serial_conn and hasattr(serial_conn, 'close'):
            try:
                serial_conn.close()
                print("🔒 Serial connection closed safely.")
            except Exception:
                pass

@app.command()
def webcam(
    source: str = typer.Option("0", "--source"), 
    conf: float = typer.Option(0.55, "--conf", "-c"), 
    req_time: float = typer.Option(3.0, "--req-time"),
    serial_enable: bool = typer.Option(False, "--serial"),
    serial_port: str = typer.Option('/dev/ttyUSB0', "--serial-port"),
    baudrate: int = typer.Option(9600, "--baud"),
    serial_map: str | None = typer.Option(None, "--serial-map"),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    typer.echo(f"Starting ZEUS Manual Render | Source: {source} | Conf: {conf}")
    process(source, conf, req_time, serial_enable, serial_port, baudrate, serial_map, debug)

if __name__ == "__main__":
    app()