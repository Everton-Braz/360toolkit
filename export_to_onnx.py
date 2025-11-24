from ultralytics import YOLO
import sys
import os

def export_model(model_path, output_name=None):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("Exporting to ONNX...")
    # Export the model to ONNX format
    # opset=12 is widely supported
    path = model.export(format="onnx", opset=12)
    
    print(f"Export complete: {path}")
    return path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default to small model if not specified
        model_path = "yolov8s-seg.pt"
        
    export_model(model_path)
