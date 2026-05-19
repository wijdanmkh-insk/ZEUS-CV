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
Path(os.path.dirname(JSON)).mkdir(parents=True, exist_ok=True)
REQ_TIME = 5.0

# Load model
model = YOLO("ref/best.pt")  
class_names = model.names
print("✅ Model loaded: ref/best.pt")
print(f"📋 Class names: {class_names}")
print(f"📊 Total classes: {len(class_names)}")
app = typer.Typer(help="YOLO detection on webcam using pure OpenCV rendering")

def save_detections(class_name: str):
    """
    Simpan deteksi ke file JSON dengan timestamp.
    Format: {"timestamp": "2026-05-15 HH:MM:SS", "class": "Organik"}
    """
    from datetime import datetime
    
    detections = []
    if os.path.exists(JSON):
        try:
            with open(JSON, 'r') as f:
                detections = json.load(f)
        except Exception:
            detections = []
    
    # Tambah deteksi baru
    detections.append({
        "timestamp": datetime.now().isoformat(),
        "class": class_name
    })
    
    # Simpan kembali
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

    # Set resolusi kamera biar enteng (opsional, tapi bagus buat live tracking)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to grab frame.")
            break

        # 1. Ambil frame copy untuk tempat ngegambar box
        annotated_frame = frame.copy()

        # 2. Run inference langsung ambil objek boxes-nya
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        
        # Ekstrak data kotak, skor, dan class id secara manual dari tensor YOLO
        boxes = results.boxes.xyxy.cpu().numpy()  # Koordinat [x1, y1, x2, y2]
        scores = results.boxes.conf.cpu().numpy() # Skor confidence
        clss = results.boxes.cls.cpu().numpy()   # ID Kelas
        
        # 🔍 DEBUG: Show detection count
        if debug or len(boxes) == 0:
            print(f"[DEBUG] Detections found: {len(boxes)} | Conf threshold: {conf_threshold}")

        # 3. Gambar Bounding Box MANUAL pakai OpenCV (Anti-Bug Supervision)
        for box, score, cls in zip(boxes, scores, clss):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            
            # Ambil nama kelas
            if isinstance(class_names, dict):
                class_name = class_names.get(class_id, str(class_id))
            else:
                class_name = class_names[class_id]

            # Set warna box (B, G, R) -> Hijau cerah buat deteksi
            color = (0, 255, 0) 
            
            # Gambar Kotak Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Bikin teks label (Nama Kelas + Score %)
            label = f"{class_name} {score:.2f}"
            
            # Gambar background teks biar kebaca
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + len(label)*10, y1), color, -1)
            # Tulis teks label di atas kotak
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # --- Logika Timer dan Serial (Tetap Berjalan Aman) ---
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

        # 4. Tampilkan live stream frame yang udah digambar manual
        cv2.imshow("ZEUS Live Cam - YOLO Manual Render", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.command()
def webcam(
    source: str = typer.Option("1", "--source"),
    conf: float = typer.Option(0.15, "--conf", "-c"), # Gw turunin default ke 15% biar sensitif
    req_time: float = typer.Option(5.0, "--req-time"),
    serial_enable: bool = typer.Option(False, "--serial"),
    serial_port: str = typer.Option('/dev/ttyUSB0', "--serial-port"),
    baudrate: int = typer.Option(115200, "--baud"),
    serial_map: str | None = typer.Option(None, "--serial-map"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show detection debug info"),
):
    typer.echo(f"Starting ZEUS Manual Render | Source: {source} | Conf: {conf}")
    process(source, conf, req_time, serial_enable, serial_port, baudrate, serial_map, debug)

if __name__ == "__main__":
    app()