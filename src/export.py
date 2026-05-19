from ultralytics import YOLO
import shutil
import os

# Download model
model = YOLO('yolo26n.pt')

# Get the local path where YOLO cached it
local_path = model.model.model_path if hasattr(model.model, 'model_path') else None

# If auto-cached, find it
if not local_path:
    model_dir = os.path.expanduser('~/.local/share/ultralytics/')
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if 'yolo26n' in f and f.endswith('.pt'):
                local_path = os.path.join(model_dir, f)
                break

# Copy to your project
if local_path:
    shutil.copy(local_path, '/model/yolo26n.pt')
    print("✅ Model saved to model/yolo26n.pt")
else:
    print("Model downloaded but path unknown. Check ~/.local/share/ultralytics/")