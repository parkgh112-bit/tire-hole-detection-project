from ultralytics import YOLO
import argparse
import os

def train_model(data_yaml, epochs=50, imgsz=1280, batch=16):
    # Load a pretrained YOLOv8 model (n, s, m, l, x)
    model = YOLO("yolov8n.pt") 

    # Start training
    results = model.train(
        data=data_yaml,       # path to data.yaml
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="tire_hole_v8",  # output folder name
        device=0,             # use GPU 0, or 'cpu'
        workers=4
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Tire Hole Dataset")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if os.path.exists(args.data):
        train_model(args.data, args.epochs)
    else:
        print(f"[Error] data.yaml not found at {args.data}. Please check your path.")
