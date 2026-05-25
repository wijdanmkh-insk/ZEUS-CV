#include <Arduino.h>
#include <Servo.h>

// =============================================================================
// Konfigurasi Servo
// =============================================================================
Servo tiltServo;
Servo panServo;

int lastTilt = 0;
int lastPan  = 0;
const int STEP_DELAY_MS = 10; // ms per langkah gerakan halus

// =============================================================================
// Konfigurasi Ultrasonik
// =============================================================================
const int   TRIG_PIN   = 2;
const int   ECHO_PIN   = 3;
const float BIN_HEIGHT = 40.0; // tinggi tong sampah dalam cm

// =============================================================================
// Buffer Serial
// =============================================================================
String inputBuffer = "";

// =============================================================================
// Posisi Servo per Kategori
// Sesuaikan nilai PAN & TILT dengan layout fisik tong sampahmu!
// =============================================================================
struct ServoPos {
  int pan;
  int tilt;
};

const ServoPos POS_HOME      = {  0,  0 };
const ServoPos POS_ANORGANIC = {  0,  0 }; // 'W' → kanan
const ServoPos POS_ORGANIC   = {179, 60 }; // 'O' → kiri
const ServoPos POS_PAPER     = {  0, 60 }; // 'P' → tengah miring

const unsigned long DUMP_HOLD_MS   = 3000; // Durasi tahan posisi (ms)
const unsigned long LOOP_DELAY_MS  = 100;  // Delay loop utama (ms)

// =============================================================================
// Helper: Gerak servo secara halus (smooth sweep)
// =============================================================================
void smoothMove(Servo &s, int fromAngle, int toAngle) {
  if (fromAngle < toAngle) {
    for (int i = fromAngle; i <= toAngle; i++) {
      s.write(i);
      delay(STEP_DELAY_MS);
    }
  } else {
    for (int i = fromAngle; i >= toAngle; i--) {
      s.write(i);
      delay(STEP_DELAY_MS);
    }
  }
}

// =============================================================================
// Helper: Pindahkan kedua servo ke posisi target
// =============================================================================
void moveServos(int pan, int tilt) {
  smoothMove(tiltServo, lastTilt, tilt);
  smoothMove(panServo,  lastPan,  pan);
  lastTilt = tilt;
  lastPan  = pan;
}

// =============================================================================
// Helper: Kembali ke posisi home dan bersihkan buffer serial
// =============================================================================
void resetToHome() {
  Serial.println("↩  Kembali ke Home Position...");
  moveServos(POS_HOME.pan, POS_HOME.tilt);

  // Buang semua byte yang masuk selama servo bergerak & delay
  while (Serial.available() > 0) {
    Serial.read();
  }
  inputBuffer = "";
  Serial.println("✅ READY — Buffer bersih, siap objek berikutnya.");
}

// =============================================================================
// Helper: Baca jarak ultrasonik dalam cm
// =============================================================================
float readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // timeout 30 ms
  return (duration * 0.0343f) / 2.0f;
}

// =============================================================================
// Helper: Konversi jarak → persentase kepenuhan tong
// =============================================================================
float calcFillPercent(float distCM) {
  if (distCM <= 0 || distCM > BIN_HEIGHT) return 0.0f;
  float pct = ((BIN_HEIGHT - distCM) / BIN_HEIGHT) * 100.0f;
  return constrain(pct, 0.0f, 100.0f);
}

// =============================================================================
// Proses Perintah Serial
// =============================================================================
void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  Serial.print("📥 Perintah diterima: '");
  Serial.print(cmd);
  Serial.println("'");

  // -----------------------------------------------------------------
  // Perintah otomatis dari Python ZEUS (byte tunggal: W / O / P)
  // -----------------------------------------------------------------
  if (cmd.length() == 1) {
    char trigger = toupper(cmd[0]); // normalisasi uppercase

    ServoPos target;
    bool valid = true;
    String label;

    switch (trigger) {
      case 'W':
        target = POS_ANORGANIC;
        label  = "ANORGANIC";
        break;
      case 'O':
        target = POS_ORGANIC;
        label  = "ORGANIC";
        break;
      case 'P':
        target = POS_PAPER;
        label  = "PAPER";
        break;
      default:
        valid = false;
        Serial.print("⚠️  Karakter tidak dikenal: '");
        Serial.print(trigger);
        Serial.println("'");
        break;
    }

    if (valid) {
      Serial.print("🎯 TRIGGER: ");
      Serial.print(label);
      Serial.print(" → PAN=");
      Serial.print(target.pan);
      Serial.print(" TILT=");
      Serial.println(target.tilt);

      // 1. Gerak ke posisi pembuangan
      moveServos(target.pan, target.tilt);

      // 2. Tahan untuk menjatuhkan sampah
      Serial.println("⏳ Menahan posisi...");
      delay(DUMP_HOLD_MS);

      // 3. Reset ke home + bersihkan buffer
      resetToHome();
    }
    return;
  }

  // -----------------------------------------------------------------
  // Perintah manual via Serial Monitor Arduino (untuk debugging)
  // -----------------------------------------------------------------
  String cmdUpper = cmd;
  cmdUpper.toUpperCase();

  if (cmdUpper == "POS") {
    Serial.print("📍 POS saat ini — PAN: ");
    Serial.print(lastPan);
    Serial.print(" | TILT: ");
    Serial.println(lastTilt);

  } else if (cmdUpper == "HOME") {
    resetToHome();

  } else if (cmdUpper == "DIST") {
    float d = readDistanceCM();
    float p = calcFillPercent(d);
    Serial.print("📏 Jarak: ");
    Serial.print(d);
    Serial.print(" cm | Penuh: ");
    Serial.print(p);
    Serial.println("%");

  } else if (cmdUpper.startsWith("PAN ")) {
    int val = cmd.substring(4).toInt();
    val = constrain(val, 0, 180);
    Serial.print("🔄 Set PAN: ");
    Serial.println(val);
    moveServos(val, lastTilt);

  } else if (cmdUpper.startsWith("TILT ")) {
    int val = cmd.substring(5).toInt();
    val = constrain(val, 0, 180);
    Serial.print("🔄 Set TILT: ");
    Serial.println(val);
    moveServos(lastPan, val);

  } else {
    Serial.print("❓ Perintah tidak dikenal: '");
    Serial.print(cmd);
    Serial.println("'");
    Serial.println("   Perintah valid: W / O / P / HOME / POS / DIST / PAN <0-180> / TILT <0-180>");
  }
}

// =============================================================================
// Setup
// =============================================================================
void setup() {
  // ⚠️ Baud rate HARUS sama dengan Python (serial_bridge.py & --baud di main.py)
  Serial.begin(115200);

  // Attach servo ke pin PWM
  tiltServo.attach(9);
  panServo.attach(10);

  // Sensor ultrasonik
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Posisi awal
  tiltServo.write(POS_HOME.tilt);
  panServo.write(POS_HOME.pan);
  lastTilt = POS_HOME.tilt;
  lastPan  = POS_HOME.pan;

  Serial.println("====================================");
  Serial.println("  ZEUS Arduino MCU — SYSTEM READY  ");
  Serial.println("  Trigger: W=Anorganic O=Organic P=Paper");
  Serial.println("====================================");
}

// =============================================================================
// Loop Utama
// =============================================================================
void loop() {
  // Baca serial non-blocking per karakter
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();

    // Karakter trigger tunggal W/O/P → langsung proses tanpa buffer
    if ((inChar == 'W' || inChar == 'w' ||
         inChar == 'O' || inChar == 'o' ||
         inChar == 'P' || inChar == 'p') && inputBuffer.length() == 0) {
      processCommand(String(inChar));
      return; // Kembali ke loop biar lebih responsif setelah proses selesai
    }
    // Newline → proses buffer (untuk perintah manual Serial Monitor)
    else if (inChar == '\n' || inChar == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    }
    // Karakter biasa → kumpulkan ke buffer
    else {
      inputBuffer += inChar;
    }
  }

  // Baca ultrasonik di background (di-comment agar tidak membanjiri serial)
  // float d = readDistanceCM();
  // float p = calcFillPercent(d);
  // Serial.print("DIST: "); Serial.print(d); Serial.print(" | %: "); Serial.println(p);

  delay(LOOP_DELAY_MS);
}
