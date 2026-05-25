import time
import os

# ⚡ PAKSA GPIOZERO PAKAI BACKEND RPi.GPIO (Anti-Error LGPIO di dalam env)
os.environ['GPIOZERO_PIN_FACTORY'] = 'rpigpio'

from gpiozero import Servo

print("🔌 Menggunakan backend: RPi.GPIO")

# --- Konfigurasi Sudut 90 Derajat ---
# Umumnya servo standar (0-180°) punya range pulse 0.5ms sampai 2.5ms.
# Agar jangkauannya HANYA 90 derajat (setengah dari total putaran), 
# kita set max_pulse_width-nya di kisaran 1.5ms (tengah-tengah).
MIN_PULSE = 0.5 / 1000  # 0 derajat
MAX_PULSE = 1.5 / 1000  # 90 derajat

# Inisialisasi Servo pada GPIO 12 (Pan) dan GPIO 13 (Tilt)
servo_pan = Servo(12, min_pulse_width=MIN_PULSE, max_pulse_width=MAX_PULSE)
servo_tilt = Servo(13, min_pulse_width=MIN_PULSE, max_pulse_width=MAX_PULSE)

try:
    print("\n🚀 Memulai Tes Jangkauan 90 Derajat...")
    print("Mekanik akan bergerak bolak-balik dari 0° ke 90° setiap 2 detik.")
    
    while True:
        # 1. Posisi 0 Derajat (Nilai -1.0 di gpiozero)
        print("➡️ Posisi: 0 Derajat (MIN)")
        servo_pan.value = -1.0
        servo_tilt.value = -1.0
        time.sleep(2)
        
        # 2. Posisi 45 Derajat (Nilai 0.0 di gpiozero, titik tengah dari range baru)
        print("⏸️ Posisi: 45 Derajat (MID)")
        servo_pan.value = 0.0
        servo_tilt.value = 0.0
        time.sleep(2)
        
        # 3. Posisi 90 Derajat (Nilai 1.0 di gpiozero, batas maksimal baru)
        print("⬅️ Posisi: 90 Derajat (MAX)")
        servo_pan.value = 1.0
        servo_tilt.value = 1.0
        time.sleep(2)

except KeyboardInterrupt:
    print("\n⏹️ Tes dihentikan oleh user.")
finally:
    # Lepas pin PWM agar servo rileks, tidak panas, dan tidak bergetar
    servo_pan.detach()
    servo_tilt.detach()
    print("🔌 Pin PWM berhasil dilepas safely.")