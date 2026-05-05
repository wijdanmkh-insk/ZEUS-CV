import cv2

# Initialize camera (0 is usually the integrated webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("ZEUS System Initialized. Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # --- ZEUS Processing Logic ---
    # 1. Pre-process (Resize for your model)
    # 2. Run Inference (Edge AI classification)
    # 3. Trigger Actuators (via Raspberry Pi / ESP32)
    
    # Example: Simple Grayscale for debugging
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('ZEUS Vision Feed', processed_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()