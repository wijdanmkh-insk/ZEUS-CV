#!/usr/bin/env python3
"""
Compare yolo26n.pt vs best.pt to diagnose detection issues.
"""
import cv2
from ultralytics import YOLO
import sys

def test_model(model_path: str, frame, conf=0.8):
    """Test a model and return detection info"""
    try:
        model = YOLO(model_path)
        print(f"\n📦 Model: {model_path}")
        print(f"📋 Classes ({len(model.names)}): {model.names}")
        
        results = model(frame, conf=conf, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
        
        print(f"✅ Detections: {len(boxes)}")
        
        for i, (box, score, cls) in enumerate(zip(boxes, scores, clss)):
            class_name = model.names[int(cls)]
            print(f"  [{i+1}] {class_name}: {score:.2%} confidence")
        
        return len(boxes) > 0
        
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return False

def main():
    # Test with sample frame
    print("🎥 Testing models with sample frame...")
    
    # Create dummy frame (or use camera)
    try:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        ret, frame = cap.read()
        if not ret:
            print("Could not grab frame. Creating dummy frame...")
            frame = cv2.imread("dataset_sampah/R/R_0001.jpg")
            if frame is None:
                print("❌ No frame available. Exiting.")
                return
        
        cap.release()
    except:
        print("Using dummy frame")
        frame = cv2.zeros((480, 640, 3), dtype='uint8')
    
    print(f"Frame size: {frame.shape}")
    
    # Compare models
    yolo26n_works = test_model("model/yolo26n.pt", frame, conf=0.15)
    best_works = test_model("model/best.pt", frame, conf=0.15)
    
    # Diagnosis
    print("\n" + "="*50)
    print("📊 DIAGNOSIS:")
    print("="*50)
    
    if yolo26n_works and not best_works:
        print("⚠️  best.pt is NOT detecting objects.")
        print("\nPossible reasons:")
        print("1. Model not trained properly")
        print("2. Confidence threshold too high")
        print("3. Model trained on different data")
        print("\n🔧 Try:")
        print("   python src/detect.py --conf 0.05 --source 1 --debug")
        print("   (Lower the confidence threshold)")
    
    elif yolo26n_works and best_works:
        print("✅ Both models work! Just need to tune best.pt")
        print("🔧 Try adjusting confidence with: --conf 0.05")
    
    elif not yolo26n_works and not best_works:
        print("❌ Neither model is detecting (camera/frame issue)")
    
    else:
        print("✅ best.pt works!")

if __name__ == "__main__":
    main()
