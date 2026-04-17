from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import time
import argparse
import os

def run_sahi_inference(model_path, image_path, conf=0.5):
    # Model Loading
    detection_model = Yolov8DetectionModel(
        model_path=model_path,
        confidence_threshold=conf,
        image_size=1280,
        device="cpu"  # default to cpu, change to "0" if GPU is available
    )

    # Load Image
    img = cv2.imread(image_path)
    if img is None: return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    start = time.time()

    # SAHI Sliced Prediction
    result = get_sliced_prediction(
        image=img_rgb,
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.35,
        overlap_width_ratio=0.35,
        postprocess_type="NMS"
    )

    end = time.time()
    print(f"Inference time: {end - start:.4f} sec")

    # Draw results
    for pred in result.object_prediction_list:
        x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAHI Inference Test")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--image", type=str, required=True, help="Path to sample image")
    args = parser.parse_args()

    if os.path.exists(args.model) and os.path.exists(args.image):
        img_result = run_sahi_inference(args.model, args.image)
        if img_result is not None:
            cv2.imshow('Detection Result', cv2.resize(img_result, (1200, 800)))
            cv2.waitKey(0)
    else:
        print("[Error] Model or image file path not valid.")
