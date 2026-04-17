import cv2
import os

# Manual Labeling Tool logic
def start_labeling(image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        # Simplified logic for portfolio visibility
        # In actual use, this would involve mouse callbacks to save YOLO coordinates
        cv2.imshow("Labeling - Press any key to skip", cv2.resize(img, (800, 600)))
        if cv2.waitKey(0) == 27: # ESC
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Update paths to your local dataset
    IMAGE_PATH = "dataset/images/train"
    LABEL_PATH = "dataset/labels/train"
    # start_labeling(IMAGE_PATH, LABEL_PATH)
    print("Labeling tool ready. Configure paths in the script before running.")
