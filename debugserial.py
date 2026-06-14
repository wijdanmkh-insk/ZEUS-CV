import serial
import time

# --- KONFIGURASI ---
# Ganti 'COM3' sesuai dengan port ESP32 kamu (di Mac/Linux biasanya '/dev/ttyUSB0' atau '/dev/ttyACM0')
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200

try:
    # Membuka koneksi serial
    esp32 = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=1)
    time.sleep(2) # Beri waktu 2 detik agar ESP32 selesai reset setelah koneksi dibuka
    print(f"Terhubung ke ESP32 di port {SERIAL_PORT}")

    while True:
        # Mengambil input dari user
        data_to_send = input("Masukkan pesan untuk ESP32 (ketik 'exit' untuk keluar): ")
        
        if data_to_send.lower() == 'exit':
            print("Menutup koneksi...")
            break
            
        # Kirim data dengan tambahan '\n' (newline) sebagai penanda akhir pesan
        # .encode('utf-8') mengubah string menjadi bytes karena serial hanya menerima bytes
        esp32.write((data_to_send + '\n').encode('utf-8'))
        print(f"-> Data terkirim: {data_to_send}")

except serial.SerialException as e:
    print(f"Error: Tidak bisa membuka port {SERIAL_PORT}. Periksa koneksi atau ganti nama port.")
except KeyboardInterrupt:
    print("\nProgram dihentikan oleh pengguna.")
finally:
    # Pastikan port ditutup kembali
    if 'esp32' in locals() and esp32.is_open:
        esp32.close()
        print("Port serial telah ditutup.")