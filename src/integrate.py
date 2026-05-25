import cv2
from ultralytics import YOLO
import typer
import time
import os
import json
from pathlib import Path
from gpiozero import Servo
# Menghapus LGPIOFactory supaya fallback ke backend native

# File configuration
JSON = os.path.join(os.path.dirname(__file__), "../res/detected.json")
CSV_LOG = os.path.join(os.path.dirname(__file__), "../res/latency_log.csv")

Path(os.path.dirname(JSON)).mkdir(parents=True, exist_ok=True)
REQ_TIME = 2.0

# ==============================================================================
# 🤖 INISIALISASI DRIVER SERVO DIRECT GPIO (RASPBERRY PI 5)
# ==============================================================================
print("🔌 Initializing Direct GPIO Servos...")
# Menghapus paksaan factory lgpio, biarkan gpiozero memilih otomatis (RPi.GPIO / rpikernel)

# Setup Servo di GPIO 12 (Tilt/Pin 9 di Arduino) dan GPIO 13 (Pan/Pin 10 di Arduino)
tilt_servo  = Servo(12, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
pan_servo = Servo(13, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Simpan state posisi dalam format internal gpiozero (-1.0 sampai 1.0)
current_pan = -1.0
current_tilt = -1.0

# ==============================================================================
# 🛠️ HELPER FUNCTIONS FOR SERVO CONTROL
# ==============================================================================
def angle_to_value(angle):
    """Mengubah derajat sudut (0-180) ke nilai internal gpiozero (-1.0 sampai 1.0)"""
    return (angle / 90.0) - 1.0

def smooth_move(servo_obj, from_val, to_val, steps=25, step_delay=0.01):
    """Menggerakkan servo secara halus per step untuk mengurangi beban mekanik/tersentak"""
    if from_val == to_val:
        servo_obj.value = to_val
        return
        
    step_size = (to_val - from_val) / steps
    actual_pos = from_val
    for _ in range(steps):
        actual_pos += step_size
        servo_obj.value = actual_pos
        time.sleep(step_delay)
    servo_obj.value = to_val

def moveServos(pan: int, tilt: int):
    """Menyamakan fungsi dari Arduino `moveServos(int pan, int tilt)` ke format Raspberry"""
    global current_pan, current_tilt
    
    target_pan = angle_to_value(pan)
    target_tilt = angle_to_value(tilt)
    
    smooth_move(tilt_servo, current_tilt, target_tilt)
    smooth_move(pan_servo, current_pan, target_pan)
    
    current_pan = target_pan
    current_tilt = target_tilt

def resetToHome():
    """Fungsi persis seperti `resetToHome()` di Arduino"""
    print("↩️  Kembali ke Home Position...")
    moveServos(pan=0, tilt=0)
    print("✅ READY — Siap mendeteksi objek berikutnya.")

def execute_sorting(pan_angle, tilt_angle):
    """Fungsi utama persis seperti di baris `Trigger Logika Arduino`"""
    
    # 1. Gerak ke posisi pembuangan
    moveServos(pan_angle, tilt_angle)
    
    # 2. Tahan posisi 3 detik sesuai logika DUMP_HOLD_MS
    print("⏳ Menahan posisi pembuangan sampah...")
    time.sleep(3.0)
    
    # 3. Reset otomatis ke posisi Home (0, 0)
    resetToHome()

# ==============================================================================
# ⚙️ INISIALISASI MODEL YOLO
# ==============================================================================
model = YOLO("./model/rpi.onnx")  
class_names = model.names
print("✅ Model loaded: model/rpi.onnx")
print(f"📋 Class names: {class_names}")

if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w") as f:
        f.write("timestamp,latency_ms,inference_fps\n")

app = typer.Typer(help="ZEUS YOLO Detection with Direct GPIO Servo Control")

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

def process(source: str, conf_threshold: float = 0.6, required_time: float = REQ_TIME, debug: bool = False):
    global current_pan, current_tilt
    
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source '{source}'.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_area = 640 * 480 

    # Set posisi awal servo ke posisi Home (0,0) saat program dimulai
    print("🏠 Set posisi awal ke Home Position (0, 0)...")
    pan_servo.value = angle_to_value(0)
    tilt_servo.value = angle_to_value(0)
    current_pan = angle_to_value(0)
    current_tilt = angle_to_value(0)

    object_timers = {}
    logged_objects = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to grab frame.")
                break

            annotated_frame = frame.copy()
            start_time = time.perf_counter()
            
            results_list = model(frame, conf=conf_threshold, imgsz=640, device="cpu", verbose=False)
            results = results_list[0]
            
            end_time = time.perf_counter()
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

                box_area = (x2 - x1) * (y2 - y1)
                if box_area > (0.70 * frame_area):
                    continue  

                current_frame_classes.add(class_name)

                # Render Bounding Box Manual
                color = (0, 255, 0) 
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {score:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + len(label)*10, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                if class_name in logged_objects:
                    cv2.putText(annotated_frame, f"LOCKED: {class_name}", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    continue

                if class_name not in object_timers:
                    object_timers[class_name] = time.time()
                    elapsed_time = 0.0
                else:
                    elapsed_time = time.time() - object_timers[class_name]
                    
                cv2.putText(annotated_frame, f"Hold: {elapsed_time:.1f}s / {required_time}s", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                # ==============================================================================
                # 🔥 LOGIKA INTEGRASI PEMBENTURAN KATEGORI SAMPAH KE SERVO GPIO
                # ==============================================================================
                if elapsed_time >= required_time:
                    save_detections(class_name)
                    logged_objects.add(class_name)
                    
                    normalized = class_name.lower().replace(' ', '_')
                    print(f"🎯 OBJECT LOCKED: Deteksi Stabil '{normalized}' Tercapai!")

                    # Mapping posisi persis seperti konfigurasi Arduino lama kamu
                    if normalized in ("anorganic", "w", "anorganic_dry", "anorganic_wet"):
                        print("🤖 Action: Moving to ANORGANIC position")
                        execute_sorting(pan_angle=0, tilt_angle=0)
                    elif normalized in ("organic", "o"):
                        print("🤖 Action: Moving to ORGANIC position")
                        execute_sorting(pan_angle=179, tilt_angle=60)
                    elif normalized in ("paper", "p"):
                        print("🤖 Action: Moving to PAPER position")
                        execute_sorting(pan_angle=0, tilt_angle=60)
                    else:
                        print(f"⚠️ Kategori '{normalized}' tidak terpetakan ke gerakan servo.")

            for active_class in list(object_timers.keys()):
                if active_class not in current_frame_classes:
                    del object_timers[active_class]

            cv2.putText(annotated_frame, f"Latency: {latency_ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Model FPS: {inf_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("ZEUS Live Cam - YOLO Direct GPIO", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Melepas pin PWM agar servo rileks saat program mati
        print("🔌 Detaching PWM Pins...")
        pan_servo.detach()
        tilt_servo.detach()
        print("🔒 Done. System shutdown safely.")

@app.command()
def webcam(
    source: str = typer.Option("0", "--source"), 
    conf: float = typer.Option(0.55, "--conf", "-c"), 
    req_time: float = typer.Option(3.0, "--req-time"),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    typer.echo(f"Starting ZEUS Direct GPIO Control | Source: {source} | Conf: {conf}")
    process(source, conf, req_time, debug)

if __name__ == "__main__":
    app()