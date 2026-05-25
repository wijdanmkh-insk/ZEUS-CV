import cv2
from ultralytics import YOLO
import typer
import time
import os
import json
from pathlib import Path

# Optional serial bridge
try:
    from src.serial_bridge import ZeusSerial
except Exception:
    try:
        from serial_bridge import ZeusSerial
    except Exception:
        ZeusSerial = None

# File configuration
JSON = os.path.join(os.path.dirname(__file__), "../res/detected.json")
# Tambahkan file log CSV untuk data latensi riset laporan ZEUS
CSV_LOG = os.path.join(os.path.dirname(__file__), "../res/latency_log.csv")

Path(os.path.dirname(JSON)).mkdir(parents=True, exist_ok=True)
REQ_TIME = 5.0

# 🛠️ PERBAIKAN 1: Hapus .to("cpu") di sini, kita akan paksa device="cpu" langsung di fungsi inferensi
model = YOLO("../model/rpi.onnx")  
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

    if serial_enable and ZeusSerial is not None:
        try:
            serial_conn = ZeusSerial(port=serial_port, baudrate=baudrate)
            print(f"Connected to MCU via {serial_port}")
        except Exception as e:
            print(f"⚠️ Could not create serial connection: {e}")
            serial_conn = None

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

            # 🛠️ PERBAIKAN 2: Tambahkan device="cpu" SECARA EKSPLISIT di sini untuk membungkam warning CUDA.
            # Catatan: Jika model ONNX kamu masih rewel soal dimensi (error index 2 got 320 expected 640), 
            # ganti imgsz=320 di bawah ini menjadi imgsz=640 atau gunakan file ONNX yang sudah di-export ulang ke 320.
            
            start_time = time.perf_counter() # Mulai hitung latensi
            
            results_list = model(frame, conf=conf_threshold, imgsz=640, device="cpu", verbose=False)
            results = results_list[0]
            
            end_time = time.perf_counter() # Selesai hitung latensi
            
            # 📊 LOGIKA LATENSI UNTUK RISET KALIBRASI
            latency_ms = (end_time - start_time) * 1000
            inf_fps = 1000 / latency_ms if latency_ms > 0 else 0
            
            # Simpan data latensi ke file CSV secara real-time
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
                    continue

                if class_name not in object_timers:
                    object_timers[class_name] = time.time()
                else:
                    elapsed_time = time.time() - object_timers[class_name]
                    if elapsed_time >= required_time:
                        save_detections(class_name)
                        logged_objects.add(class_name)
                        
                        send_category = None
                        if class_map and class_name in class_map:
                            send_category = class_map[class_name]
                        else:
                            normalized = class_name.lower().replace(' ', '_')
                            if normalized in ("organic", "anorganic", "hazard", "paper"):
                                send_category = normalized

                        if serial_conn and send_category:
                            serial_conn.send_trigger(send_category)

            # Reset Timer untuk Objek yang Hilang
            for active_class in list(object_timers.keys()):
                if active_class not in current_frame_classes:
                    del object_timers[active_class]

            # 🛠️ PERBAIKAN 3: Tampilkan info Latensi Riset langsung di Layar Video (Warna Kuning Taktis)
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
            serial_conn.close()

@app.command()
def webcam(
    source: str = typer.Option("0", "--source"), 
    conf: float = typer.Option(0.55, "--conf", "-c"), 
    req_time: float = typer.Option(5.0, "--req-time"),
    serial_enable: bool = typer.Option(False, "--serial"),
    serial_port: str = typer.Option('/dev/ttyUSB0', "--serial-port"),
    baudrate: int = typer.Option(115200, "--baud"),
    serial_map: str | None = typer.Option(None, "--serial-map"),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    typer.echo(f"Starting ZEUS Manual Render | Source: {source} | Conf: {conf}")
    process(source, conf, req_time, serial_enable, serial_port, baudrate, serial_map, debug)

if __name__ == "__main__":
    app()