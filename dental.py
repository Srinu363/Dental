import os
import shutil
import random
import yaml
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

# Set seed for reproducibility
SEED = 42
random.seed(SEED)

# Paths - adjust if needed
DATASET_DIR = "ToothNumber_TaskDataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
OUTPUT_DIR = "datasets_split"

# FDI tooth classes as per your task 
FDI_CLASSES = {
    0: "Canine (13)",
    1: "Canine (23)",
    2: "Canine (33)",
    3: "Canine (43)",
    4: "Central Incisor (21)",
    5: "Central Incisor (41)",
    6: "Central Incisor (31)",
    7: "Central Incisor (11)",
    8: "First Molar (16)",
    9: "First Molar (26)",
    10: "First Molar (36)",
    11: "First Molar (46)",
    12: "First Premolar (14)",
    13: "First Premolar (34)",
    14: "First Premolar (44)",
    15: "First Premolar (24)",
    16: "Lateral Incisor (22)",
    17: "Lateral Incisor (32)",
    18: "Lateral Incisor (42)",
    19: "Lateral Incisor (12)",
    20: "Second Molar (17)",
    21: "Second Molar (27)",
    22: "Second Molar (37)",
    23: "Second Molar (47)",
    24: "Second Premolar (15)",
    25: "Second Premolar (25)",
    26: "Second Premolar (35)",
    27: "Second Premolar (45)",
    28: "Third Molar (18)",
    29: "Third Molar (28)",
    30: "Third Molar (38)",
    31: "Third Molar (48)"
}

def remove_unmatched_images_and_labels(images_dir, labels_dir):
    images = os.listdir(images_dir)
    removed_img = 0
    for img in images:
        stem, _ = os.path.splitext(img)
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if not os.path.exists(label_path):
            os.remove(os.path.join(images_dir, img))
            removed_img += 1
    print(f"Removed {removed_img} images without labels.")

    labels = os.listdir(labels_dir)
    removed_label = 0
    valid_ext = {'.jpg', '.jpeg', '.png'}
    for label in labels:
        stem, _ = os.path.splitext(label)
        if not any(os.path.exists(os.path.join(images_dir, stem + ext)) for ext in valid_ext):
            os.remove(os.path.join(labels_dir, label))
            removed_label += 1
    print(f"Removed {removed_label} labels without images.")

def fix_bad_label_lines(labels_dir):
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    bad_lines_total = 0
    empty_label_files = 0
    for label_file in label_files:
        good_lines = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:   # class_id + 4 coords
                    good_lines.append(line.strip())
                else:
                    bad_lines_total += 1
        if good_lines:
            with open(label_file, 'w') as f:
                f.write("\n".join(good_lines) + "\n")
        else:
            os.remove(label_file)
            empty_label_files += 1
    print(f"Removed {bad_lines_total} malformed lines from labels.")
    print(f"Deleted {empty_label_files} label files with no valid lines.")

def create_splits(images_dir, labels_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    image_files = sorted(os.listdir(images_dir))
    valid_pairs = []
    for img in image_files:
        stem, _ = os.path.splitext(img)
        label_file = f"{stem}.txt"
        if os.path.isfile(os.path.join(labels_dir, label_file)):
            valid_pairs.append((img, label_file))

    random.shuffle(valid_pairs)
    n_total = len(valid_pairs)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    splits = {
        'train': valid_pairs[:n_train],
        'val': valid_pairs[n_train:n_train + n_val],
        'test': valid_pairs[n_train + n_val:]
    }

    for split_name, pairs in splits.items():
        for img_file, label_file in pairs:
            shutil.copy(os.path.join(images_dir, img_file),
                        os.path.join(output_dir, split_name, "images", img_file))
            shutil.copy(os.path.join(labels_dir, label_file),
                        os.path.join(output_dir, split_name, "labels", label_file))

    print(f"Dataset split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

def create_data_yaml(output_dir, classes_dict):
    data_yaml = {
        'train': os.path.abspath(os.path.join(output_dir, "train/images")),
        'val': os.path.abspath(os.path.join(output_dir, "val/images")),
        'test': os.path.abspath(os.path.join(output_dir, "test/images")),
        'nc': len(classes_dict),
        'names': [classes_dict[i] for i in range(len(classes_dict))]
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path

def visualize_sample_predictions(model, dataset_dir, split="test", num_samples=4):
    import matplotlib.patches as patches
    images_path = os.path.join(dataset_dir, split, "images")
    image_files = os.listdir(images_path)
    if len(image_files) == 0:
        print("No images in test split for visualization")
        return
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    fig, axs = plt.subplots(1, len(samples), figsize=(15, 5))
    if len(samples) == 1:
        axs = [axs]

    for ax, img_file in zip(axs, samples):
        img_path = os.path.join(images_path, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = model(img_path)

        ax.imshow(img_rgb)
        ax.set_title(img_file)
        ax.axis('off')

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())

            if conf > 0.5:
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                label = f"{FDI_CLASSES[cls].split('(')[1][:-1]}: {conf:.2f}"
                ax.text(x1, y1 - 10, label, color='yellow', fontsize=8, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.7, pad=1))

    plt.tight_layout()
    plt.show()

def main():
    print("Starting dataset cleanup and preparation...")
    remove_unmatched_images_and_labels(IMAGES_DIR, LABELS_DIR)
    fix_bad_label_lines(LABELS_DIR)

    print("\nSplitting dataset...")
    create_splits(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)

    print("\nCreating YAML data file...")
    data_yaml_path = create_data_yaml(OUTPUT_DIR, FDI_CLASSES)

    print("\nLoading YOLOv8 pretrained model...")
    model = YOLO('yolov8s.pt')

    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    print("\nStarting training...")
    model.train(data=data_yaml_path, epochs=100, imgsz=640, batch=16,
                device=device,
                project="runs", name="tooth_detection", exist_ok=True)

    print("Training completed.")

    print("\nEvaluating model...")
    results = model.val(data=data_yaml_path)
    print(results)

    print("\nVisualizing sample predictions from test set...")
    visualize_sample_predictions(model, OUTPUT_DIR, split="test")

if __name__ == "__main__":
    main()
