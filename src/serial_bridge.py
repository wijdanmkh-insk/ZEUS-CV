import time

try:
    import serial as _pyserial
except Exception as e:
    _pyserial = None

class ZeusSerial:
    """
    Serial bridge antara Raspberry Pi (Python) dan MCU (Arduino Nano).
    Mengirim trigger byte berdasarkan kategori sampah yang terdeteksi YOLO.
    """
    CATEGORY_MAP = {
        "anorganic":     b'W',
        "anorganic_wet": b'W',
        "anorganic_dry": b'W',
        "w":             b'W',
        "organic":       b'O',
        "o":             b'O',
        "paper":         b'P',
        "p":             b'P',
        "hazard":        b'W',
    }

    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.ser = None

        if _pyserial is None:
            print("❌ pyserial tidak tersedia. Install dengan: pip install pyserial")
            return

        try:
            self.ser = _pyserial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Menunggu Arduino kelar auto-reset
            print(f"✅ Serial terhubung: {port} @ {baudrate} baud")
        except Exception as e:
            print(f"❌ Serial Error: {e}")
            self.ser = None

    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def send_trigger(self, category: str):
        if not self.is_connected():
            print("⚠️  Serial tidak terhubung — pengiriman dilewati.")
            return

        key = category.lower().strip().replace(' ', '_')
        cmd = self.CATEGORY_MAP.get(key)

        if cmd:
            try:
                self.ser.write(cmd)
                self.ser.flush()  # Paksa kirim detik ini juga
                print(f"📡 Serial Sent: '{category}' → key='{key}' → byte=b'{cmd.decode()}'")
            except Exception as e:
                print(f"❌ Gagal menulis ke serial: {e}")
        else:
            print(f"⚠️  Tidak ada mapping untuk kategori: '{category}' (key='{key}')")