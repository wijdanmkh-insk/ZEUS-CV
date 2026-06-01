# CONTOH DI PYTHON: Pastikan ada '\n' di akhir string!
import serial
import time

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1) # Sesuaikan port ESP32-mu

def kirim_perintah(command):
    print(f"Sending -> {command}")
    # WAJIB tambahkan \n agar dibaca oleh c == '\n' di ESP32
    ser.write(f"{command}\n".encode('utf-8')) 

# Test kirim
time.sleep(2)
kirim_perintah("organic")