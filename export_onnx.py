from ultralytics import YOLO
import sys

models = ['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt']

for model_name in models:
    try:
        print(f"Exporting {model_name}...")
        model = YOLO(model_name)
        model.export(format='onnx', simplify=True)
        print(f"Successfully exported {model_name} to ONNX")
    except Exception as e:
        print(f"Failed to export {model_name}: {e}")
