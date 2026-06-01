# CONTOH DI PYTHON: Pastikan ada '\n' di akhir string!
import serial
import time

port = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
arduino = serial.Serial(f'{port}', 115200, timeout=1)
time.sleep(2)
print("✅ Arduino connected")

def kirim_perintah(command):
    print(f"Sending -> {command}")
    # WAJIB tambahkan \n agar dibaca oleh c == '\n' di ESP32
    arduino.write(f"{command}\n".encode('utf-8')) 

# Test kirim
time.sleep(2)
kirim_perintah("anorganic")