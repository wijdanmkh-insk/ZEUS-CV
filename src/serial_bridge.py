import time

try:
    import serial as _pyserial
except Exception as e:
    _pyserial = None


class ZeusSerial:
    """
    Serial bridge antara Raspberry Pi (Python) dan MCU (Arduino).
    Mengirim trigger byte berdasarkan kategori sampah yang terdeteksi YOLO.
    """

    # =========================================================================
    # Mapping kategori YOLO → byte perintah Arduino
    # PENTING: semua key HARUS lowercase agar cocok dengan hasil .lower()
    # =========================================================================
    CATEGORY_MAP = {
        # Anorganic dan sub-kelasnya → 'W'
        "anorganic":     b'W',
        "anorganic_wet": b'W',
        "anorganic_dry": b'W',
        "w":             b'W',

        # Organic → 'O'
        "organic": b'O',
        "o":       b'O',

        # Paper → 'P'
        "paper": b'P',
        "p":     b'P',

        # Hazard (belum ada posisi servo, default ke anorganic dulu)
        "hazard": b'W',
    }

    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.ser = None

        if _pyserial is None:
            print("❌ pyserial tidak tersedia. Install dengan: pip install pyserial")
            return

        try:
            self.ser = _pyserial.Serial(port, baudrate, timeout=1)
            # Delay 2 detik biar MCU selesai reset setelah port dibuka
            time.sleep(2)
            print(f"✅ Serial terhubung: {port} @ {baudrate} baud")
        except Exception as e:
            print(f"❌ Serial Error: {e}")
            self.ser = None

    def is_connected(self) -> bool:
        """Cek apakah koneksi serial aktif."""
        return self.ser is not None and self.ser.is_open

    def send_trigger(self, category: str):
        """
        Kirim byte perintah ke Arduino berdasarkan kategori sampah.

        Args:
            category: Nama kelas dari model YOLO (case-insensitive).
        """
        if not self.is_connected():
            print("⚠️  Serial tidak terhubung — pengiriman dilewati.")
            return

        # Normalisasi key: lowercase, strip spasi, spasi tengah → underscore
        key = category.lower().strip().replace(' ', '_')

        cmd = self.CATEGORY_MAP.get(key)

        if cmd:
            try:
                self.ser.write(cmd)
                # Flush paksa agar byte langsung keluar dari buffer OS ke kabel
                self.ser.flush()
                print(f"📡 Serial Sent: '{category}' → key='{key}' → byte=b'{cmd.decode()}'")
            except Exception as e:
                print(f"❌ Gagal menulis ke serial: {e}")
        else:
            print(f"⚠️  Tidak ada mapping untuk kategori: '{category}' (key='{key}')")
            print(f"    Mapping yang tersedia: {list(self.CATEGORY_MAP.keys())}")

    def close(self):
        """Tutup koneksi serial dengan aman."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
                print("🔒 Koneksi serial ditutup.")
            except Exception as e:
                print(f"⚠️  Error saat menutup serial: {e}")