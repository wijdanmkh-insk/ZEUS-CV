from gpiozero import Servo
from time import sleep

# Menggunakan Factory default (bisa disesuaikan jika gerakan servo kurang maksimal)
# Kita hubungkan kabel sinyal ke GPIO 17
servo = Servo(17)

print("Memulai pergerakan servo...")

try:
    while True:
        # Gerakkan ke posisi minimum (-1)
        print("Posisi Minimum")
        servo.min()
        sleep(1)
        
        # Gerakkan ke posisi tengah (0)
        print("Posisi Tengah")
        servo.mid()
        sleep(1)
        
        # Gerakkan ke posisi maksimum (1)
        print("Posisi Maksimum")
        servo.max()
        sleep(1)
        
        # Contoh menggerakkan perlahan dari min ke max
        print("Berputar perlahan...")
        for i in range(-10, 11):
            servo.value = i / 10.0
            sleep(0.1)
        sleep(1)

except KeyboardInterrupt:
    # Program berhenti jika Anda menekan Ctrl+C
    print("\nProgram dihentikan oleh pengguna.")