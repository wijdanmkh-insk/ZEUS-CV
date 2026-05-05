from ultralytics import YOLO

# This line downloads 'yolo26n.pt' automatically if it doesn't exist
model = YOLO("yolo26n.pt") 

# This creates 'yolo26n.onnx' in your current directory
model.export(format="onnx")