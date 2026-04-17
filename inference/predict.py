import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
import os

class TireHoleDetector:
    def __init__(self, model_path):
        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def compute_mean_brightness(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def adaptive_clahe(self, img, mu=None, T_low=40, T_high=70, clip_min=0.3, clip_max=4.0):
        """
        Custom Adaptive CLAHE logic for robust hole detection under various lighting.
        """
        if mu is None:
            mu = self.compute_mean_brightness(img)
        
        # Calculate strength based on brightness
        strength = 1.0 if mu <= T_low else 0.0 if mu >= T_high else 1 - (mu - T_low) / (T_high - T_low)
        clip_limit = clip_min + (clip_max - clip_min) * strength
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def predict(self, image_path, conf=0.25):
        img = cv2.imread(image_path)
        if img is None: return None

        # 1. Preprocessing: Apply Adaptive CLAHE
        processed_img = self.adaptive_clahe(img)

        # 2. Inference
        results = self.model.predict(processed_img, conf=conf, verbose=False)
        
        # 3. Visualization & Result Collection
        annotated_img = results[0].plot()
        
        centers = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append((cx, cy))
            # Mark center point
            cv2.circle(annotated_img, (cx, cy), 5, (0, 0, 255), -1)

        return annotated_img, centers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tire Hole Detection Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()

    detector = TireHoleDetector(args.model)
    result_img, centers = detector.predict(args.image)

    if result_img is not None:
        print(f"Found {len(centers)} holes. Centers: {centers}")
        cv2.imshow("Detection Result", cv2.resize(result_img, (1200, 800)))
        cv2.waitKey(0)
    else:
        print("[Error] Failed to process image.")
