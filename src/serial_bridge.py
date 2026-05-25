import time

try:
    import serial as _pyserial
except Exception as e:
    _pyserial = None


class ZeusSerial:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        self.ser = None
        if _pyserial is None:
            print("❌ pyserial not available (install with 'pip install pyserial')")
            return

        try:
            # Menggunakan timeout agar fungsi write/read tidak membuat program utama hang
            self.ser = _pyserial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Delay 2 detik untuk memberikan waktu MCU reset setelah serial dibuka
            print(f"✅ Connected to serial device on {port} @ {baudrate}")
        except Exception as e:
            print(f"❌ Serial Error: {e}")
            self.ser = None

    def send_trigger(self, category: str):
        if not self.ser:
            print("⚠️ Serial not connected — skipping send")
            return

        # Kamus mapping wajib menggunakan huruf kecil semua (lowercase)
        mapping = {
            "anorganic": b'W',
            "organic": b'O',
            "paper": b'P',
            
            # Amunisi tambahan jika model YOLO mengirimkan sub-kelas atau inisial tunggal
            "anorganic_wet": b'W',
            "anorganic_dry": b'W',
            "o": b'O',
            "p": b'P'
        }

        # Bersihkan string: hapus spasi gaib di ujung, kecilkan huruf, ganti spasi tengah dengan underscore
        key = category.lower().strip().replace(' ', '_')

        # Mencari perintah byte berdasarkan key yang sudah bersih
        cmd = mapping.get(key)
        
        if cmd:
            try:
                self.ser.write(cmd)
                # Flush memastikan data benar-benar terkirim keluar dari buffer RPi ke kabel
                self.ser.flush() 
                print(f"📡 Serial Sent: '{category}' -> Key: '{key}' -> Triggered: ({cmd.decode()})")
            except Exception as e:
                print(f"❌ Failed to write to serial: {e}")
        else:
            print(f"⚠️ No serial mapping for category: '{category}' (Processed key: '{key}')")