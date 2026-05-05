import cv2
import supervision as sv
from ultralytics import YOLO
import typer
import time
import os
import json

# File
JSON = "detected.json"
REQ_TIME = 5.0

# Load the model
model = YOLO("yolo26n.pt")  # Automatically downloads if not present
class_names = model.names

def save_detections(detections):
    data = [];
    if os.pat.exists(JSON):
        with open(JSON, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

    entry = {
        "object_name": detections,
        "added": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    data.append(entry)

    with open(JSON, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved detections: {detections} at {entry['added']}")

# Print class names once
print("Class names:", class_names)

app = typer.Typer(help="YOLO detection on webcam or video")

def process(source: str, output_file: str, conf_threshold: float = 0.6):
    # Open source: int for webcam, str for file
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open source '{source}'.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Fallback for webcam

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to grab frame.")
            break

        # Run inference with custom confidence
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Annotate
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Write to output
        out.write(annotated_frame)

        # Display
        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output saved to: {output_file}")

@app.command()
def webcam(
    source: str = typer.Option("0", "--source", help="0 for webcam, 1 for external, or http://ip:port/video for mobile"),
    output_file: str = typer.Option("output.mp4", "--output-file", "-o", help="Output video file"),
    conf: float = typer.Option(0.25, "--conf", "-c", help="Confidence threshold (lower = more detections, less accurate)"),
):
    typer.echo(f"Starting detection | Source: {source} | Conf: {conf} | Output: {output_file}")
    process(source, output_file, conf)

if __name__ == "__main__":
    app()