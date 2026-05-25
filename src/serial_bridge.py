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
            self.ser = _pyserial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # allow device to reset
            print(f"✅ Connected to serial device on {port} @ {baudrate}")
        except Exception as e:
            print(f"❌ Serial Error: {e}")
            self.ser = None

    def send_trigger(self, category: str):
        if not self.ser:
            print("⚠️ Serial not connected — skipping send")
            return

        mapping = {
            "Anorganic": b'W',
            "Organic": b'O',
            "Paper": b'P'
        }

        key = category.lower().replace(' ', '_')
        cmd = mapping.get(key)
        if cmd:
            try:
                self.ser.write(cmd)
                print(f"📡 Serial Sent: {category} ({cmd.decode()})")
            except Exception as e:
                print(f"❌ Failed to write to serial: {e}")
        else:
            print(f"⚠️ No serial mapping for category: {category}")
