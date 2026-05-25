import cv2
from ultralytics import YOLO
import typer
import time
import os
import json
from pathlib import Path

# ==============================================================================
# Import ZeusSerial dengan fallback yang informatif
# ==============================================================================
ZeusSerial = None
_import_error_msg = ""

try:
    from src.serial_bridge import ZeusSerial
    print("✅ Imported ZeusSerial dari src.serial_bridge")
except Exception as e1:
    try:
        from serial_bridge import ZeusSerial
        print("✅ Imported ZeusSerial dari serial_bridge")
    except Exception as e2:
        _import_error_msg = (
            f"\n   - src.serial_bridge : {e1}"
            f"\n   - serial_bridge     : {e2}"
        )
        ZeusSerial = None

# ==============================================================================
# Path Konfigurasi
# ==============================================================================
_BASE = os.path.dirname(__file__)
JSON_PATH = os.path.join(_BASE, "../res/detected.json")
CSV_LOG   = os.path.join(_BASE, "../res/latency_log.csv")

Path(os.path.dirname(JSON_PATH)).mkdir(parents=True, exist_ok=True)

DEFAULT_REQ_TIME = 2.0

# ==============================================================================
# Inisialisasi Model YOLO
# ==============================================================================
model = YOLO("./model/rpi.onnx")
class_names = model.names
print("✅ Model loaded: model/rpi.onnx")
print(f"📋 Class names  : {class_names}")
print(f"📊 Total classes: {len(class_names)}")

# Tulis header CSV jika belum ada
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w") as f:
        f.write("timestamp,latency_ms,inference_fps\n")

app = typer.Typer(help="ZEUS — YOLO Waste Detection with Serial Bridge")


# ==============================================================================
# Helper: Simpan deteksi ke JSON
# ==============================================================================
def save_detections(class_name: str):
    from datetime import datetime

    detections = []
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r") as f:
                detections = json.load(f)
        except Exception:
            detections = []

    detections.append({
        "timestamp": datetime.now().isoformat(),
        "class": class_name,
    })

    try:
        with open(JSON_PATH, "w") as f:
            json.dump(detections, f, indent=2)
        print(f"[SAVED] '{class_name}' → {JSON_PATH}")
    except Exception as e:
        print(f"⚠️  Error menyimpan deteksi: {e}")


# ==============================================================================
# Helper: Inisialisasi koneksi serial dengan debug lengkap
# ==============================================================================
def init_serial(
    serial_enable: bool,
    serial_port: str,
    baudrate: int,
) -> object | None:

    print("\n" + "=" * 60)
    print("[SERIAL INIT] Memulai inisialisasi koneksi serial...")
    print(f"  Flag --serial   : {serial_enable}")
    print(f"  ZeusSerial kelas: {'TERSEDIA' if ZeusSerial else 'TIDAK TERSEDIA'}")

    if not serial_enable:
        print("[SERIAL INIT] Serial tidak diaktifkan, skip.")
        print("=" * 60 + "\n")
        return None

    if ZeusSerial is None:
        print("❌ BATAL: Modul serial_bridge gagal di-import!")
        print(f"   Detail:{_import_error_msg}")
        print("   💡 Pastikan pyserial terinstall: pip install pyserial")
        print("=" * 60 + "\n")
        return None

    if not os.path.exists(serial_port):
        print(f"❌ BATAL: Port '{serial_port}' tidak ditemukan di sistem!")
        print("   💡 Coba: ls /dev/tty* | grep -E 'USB|ACM'")
        print("=" * 60 + "\n")
        return None

    try:
        print(f"  Membuka port {serial_port} @ {baudrate} baud...")
        conn = ZeusSerial(port=serial_port, baudrate=baudrate)

        if conn.is_connected():
            print(f"✅ KONEKSI BERHASIL: {serial_port} @ {baudrate} baud")
        else:
            print("❌ KONEKSI GAGAL: Port terbuka tapi serial tidak aktif.")
            conn = None
    except Exception as e:
        err = str(e)
        print(f"❌ KONEKSI GAGAL: {err}")

        if "Permission denied" in err:
            print(f"   💡 Fix: sudo chmod 666 {serial_port}")
        elif "Device or resource busy" in err:
            print(f"   💡 Fix: sudo lsof | grep {os.path.basename(serial_port)}")

        import traceback
        traceback.print_exc()
        conn = None

    print("=" * 60 + "\n")
    return conn


# ==============================================================================
# Core: Proses deteksi
# ==============================================================================
def process(
    source: str,
    conf_threshold: float = 0.6,
    required_time: float = DEFAULT_REQ_TIME,
    serial_enable: bool = False,
    serial_port: str = "/dev/ttyUSB0",
    baudrate: int = 115200,          # ← default 115200, sinkron dengan Arduino
    mapping_path: str | None = None,
    debug: bool = False,
):
    # Konversi source ke int jika angka (index kamera)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # Muat mapping custom jika ada
    class_map: dict | None = None
    if mapping_path:
        try:
            with open(mapping_path, "r") as mf:
                class_map = json.load(mf)
            print(f"📂 Loaded serial mapping dari: {mapping_path}")
        except Exception as e:
            print(f"⚠️  Gagal load mapping file: {e}")

    # Inisialisasi serial
    serial_conn = init_serial(serial_enable, serial_port, baudrate)

    # Buka kamera / video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka source: '{source}'")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_area = 640 * 480

    object_timers: dict[str, float] = {}
    logged_objects: set[str] = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream atau gagal baca frame.")
                break

            annotated_frame = frame.copy()

            # --- Inferensi YOLO ---
            t0 = time.perf_counter()
            results = model(frame, conf=conf_threshold, imgsz=640, device="cpu", verbose=False)[0]
            latency_ms = (time.perf_counter() - t0) * 1000
            inf_fps = 1000 / latency_ms if latency_ms > 0 else 0

            # Log latensi ke CSV
            with open(CSV_LOG, "a") as f:
                f.write(f"{time.time():.3f},{latency_ms:.2f},{inf_fps:.1f}\n")

            boxes  = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            clss   = results.boxes.cls.cpu().numpy()

            if debug and len(boxes) > 0:
                print(f"[DEBUG] {len(boxes)} deteksi | Latency: {latency_ms:.1f} ms")

            current_frame_classes: set[str] = set()

            for box, score, cls in zip(boxes, scores, clss):
                x1, y1, x2, y2 = map(int, box)
                class_id   = int(cls)
                class_name = (
                    class_names.get(class_id, str(class_id))
                    if isinstance(class_names, dict)
                    else class_names[class_id]
                )

                # Filter kotak yang terlalu besar (>70% frame) — biasanya false positive
                if (x2 - x1) * (y2 - y1) > 0.70 * frame_area:
                    continue

                current_frame_classes.add(class_name)

                # Gambar bounding box
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {score:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + len(label) * 10, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Jika sudah pernah diproses, tampilkan LOCKED dan skip
                if class_name in logged_objects:
                    cv2.putText(annotated_frame, f"LOCKED: {class_name}", (x1, y2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    continue

                # Timer hold
                if class_name not in object_timers:
                    object_timers[class_name] = time.time()
                elapsed = time.time() - object_timers[class_name]

                cv2.putText(annotated_frame, f"Hold: {elapsed:.1f}s / {required_time}s",
                            (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                # Trigger jika sudah cukup lama
                if elapsed >= required_time:
                    save_detections(class_name)
                    logged_objects.add(class_name)

                    # Tentukan kategori yang akan dikirim ke Arduino
                    send_category: str | None = None

                    if class_map and class_name in class_map:
                        send_category = class_map[class_name]
                    else:
                        # Normalisasi langsung dari nama kelas model
                        norm = class_name.lower().replace(" ", "_")
                        valid = {"organic", "anorganic", "anorganic_wet", "anorganic_dry", "paper", "hazard", "o", "p", "w"}
                        if norm in valid:
                            send_category = norm

                    if serial_conn and send_category:
                        try:
                            serial_conn.send_trigger(send_category)
                        except Exception as e_send:
                            print(f"⚠️  Gagal kirim serial: {e_send}")
                    elif not serial_conn and serial_enable:
                        print(f"⚠️  Serial tidak aktif, '{class_name}' tidak dikirim.")

            # Reset timer untuk objek yang sudah tidak terdeteksi
            for cls_name in list(object_timers.keys()):
                if cls_name not in current_frame_classes:
                    del object_timers[cls_name]

            # Overlay info latensi
            cv2.putText(annotated_frame, f"Latency : {latency_ms:.1f} ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Inf FPS : {inf_fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("ZEUS Live Cam — YOLO Waste Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("\n🧹 Membersihkan resources...")
        cap.release()
        cv2.destroyAllWindows()
        if serial_conn:
            serial_conn.close()


# ==============================================================================
# CLI Entry Point
# ==============================================================================
@app.command()
def webcam(
    source: str = typer.Option("0", "--source", "-s",
                               help="Index kamera (0,1,...) atau path video file"),
    conf: float = typer.Option(0.55, "--conf", "-c",
                               help="Threshold confidence deteksi (0.0–1.0)"),
    req_time: float = typer.Option(3.0, "--req-time",
                                   help="Durasi hold (detik) sebelum trigger dikirim"),
    serial_enable: bool = typer.Option(False, "--serial",
                                       help="Aktifkan komunikasi serial ke Arduino"),
    serial_port: str = typer.Option("/dev/ttyUSB0", "--serial-port",
                                    help="Port serial Arduino (misal /dev/ttyACM0)"),
    baudrate: int = typer.Option(115200, "--baud",          # ← 115200, sama dengan Arduino
                                 help="Baud rate serial (harus sama dengan Arduino)"),
    serial_map: str | None = typer.Option(None, "--serial-map",
                                          help="Path ke JSON mapping kategori → serial"),
    debug: bool = typer.Option(False, "--debug", "-d",
                               help="Tampilkan info debug di terminal"),
):
    typer.echo(f"🚀 ZEUS Start | Source: {source} | Conf: {conf} | Baud: {baudrate}")
    process(source, conf, req_time, serial_enable, serial_port, baudrate, serial_map, debug)


if __name__ == "__main__":
    app()