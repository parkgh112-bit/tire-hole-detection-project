import os
import cv2
import albumentations as A
import numpy as np

# 1. Augmentation Pipeline Definition
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),
    A.ISONoise(color_shift=(0, 0.05), intensity=(0.1, 0.3), p=0.2),
    A.MotionBlur(blur_limit=3, p=0.1),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.1),
    A.Rotate(limit=10, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_yolo_labels(label_path):
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5: continue
        try:
            class_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            bboxes.append([x, y, w, h])
            class_labels.append(class_id)
        except ValueError: continue
    return bboxes, class_labels

def save_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for cls, (x, y, w, h) in zip(class_labels, bboxes):
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def augment_and_save(image_path, label_path, save_dir, num_aug=5):
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)
    img = cv2.imread(image_path)
    if img is None: return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = read_yolo_labels(label_path)
    if not bboxes: return
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for i in range(num_aug):
        try:
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        except: continue
        aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        img_save_path = os.path.join(save_dir, 'images', f"{base_name}_aug{i}.jpg")
        label_save_path = os.path.join(save_dir, 'labels', f"{base_name}_aug{i}.txt")
        cv2.imwrite(img_save_path, aug_img)
        save_yolo_labels(label_save_path, augmented['bboxes'], augmented['class_labels'])

if __name__ == "__main__":
    # Update these paths for your environment
    image_dir = "path/to/your/images"
    label_dir = "path/to/your/labels"
    save_dir = "output_aug"
    num_aug = 1
    # Add your processing loop here
