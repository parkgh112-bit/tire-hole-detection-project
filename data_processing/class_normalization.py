import os
import argparse

def normalize_yolo_labels(label_dir):
    """
    Normalizes YOLO label IDs (e.g., 0.0 -> 0) and ensures correct format.
    """
    if not os.path.exists(label_dir):
        print(f"[Error] Label directory not found: {label_dir}")
        return

    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            path = os.path.join(label_dir, filename)
            with open(path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    # Convert class ID float -> int -> str (e.g., 0.0 -> 0)
                    parts[0] = str(int(float(parts[0])))
                    new_lines.append(" ".join(parts) + "\n")

            with open(path, "w") as f:
                f.writelines(new_lines)
    print(f"Finished normalization for {label_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize YOLO Label Classes")
    parser.add_argument("--dir", type=str, default="dataset/labels/train", help="Label directory")
    args = parser.parse_args()

    # To run: python data_processing/class_normalization.py --dir 전송용/
    normalize_yolo_labels(args.dir)
