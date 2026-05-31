import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

ret, background = cap.read()

if not ret:
    print("Gagal ambil background")
    exit()

background = cv2.resize(background, (640, 480))
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

# ======================
# WINDOW + TRACKBAR
# ======================

cv2.namedWindow("Controls")

cv2.createTrackbar(
    "Threshold",
    "Controls",
    30,
    255,
    nothing
)

cv2.createTrackbar(
    "Min Area",
    "Controls",
    5000,
    50000,
    nothing
)

object_count = 0
object_present = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (21, 21),
        0
    )

    # ======================
    # READ TRACKBARS
    # ======================

    threshold_value = cv2.getTrackbarPos(
        "Threshold",
        "Controls"
    )

    min_area = cv2.getTrackbarPos(
        "Min Area",
        "Controls"
    )

    # ======================
    # DIFFERENCE
    # ======================

    diff = cv2.absdiff(
        background_gray,
        gray
    )

    _, thresh = cv2.threshold(
        diff,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    thresh = cv2.dilate(
        thresh,
        None,
        iterations=2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = False

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        detected = True

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{int(area)}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # ======================
    # OBJECT COUNTER
    # ======================

    if detected and not object_present:
        object_count += 1
        print(f"Object #{object_count}")

    object_present = detected

    cv2.putText(
        frame,
        f"Count: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", thresh)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    # tekan R untuk update background
    if key == ord('r'):
        background_gray = gray.copy()
        print("Background Updated")

cap.release()
cv2.destroyAllWindows()