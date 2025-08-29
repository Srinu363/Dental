# Dental Tooth Detection with YOLOv8 Using FDI Numbering System

---

![Sample Dental X-ray](https://github.com/Srinu363/Dental/blob/master/runs/tooth_detection/train_batch0.jpg)
*Figure 1: Representative Panoramic Dental X-ray Image from Dataset*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [FDI Tooth Numbering System Explanation](#fdi-tooth-numbering-system-explanation)
- [Project Workflow](#project-workflow)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Training Details and Parameters](#training-details-and-parameters)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Results and Visualizations](#results-and-visualizations)
- [Post-Processing Logic (Optional)](#post-processing-logic-optional)
- [Project Structure](#project-structure)


---

## Project Overview

This project presents a deep learning approach to automatic tooth detection and identification on dental panoramic radiographs through the YOLOv8 object detection framework. Each tooth is localized and classified into one of 32 categories defined by the widely used FDI numbering system, facilitating dental diagnostic workflows and research initiatives.

**The core goals are:**
- Precise bounding box detection for individual teeth.
- Accurate multi-class classification using the FDI system.
- Development of a robust pipeline for dataset preparation, model training, evaluation, visualization, and optional post-processing corrections.
- End-to-end implementation suitable for local machines and reproducible in research settings.

---

## Problem Statement

Manual tooth identification on panoramic dental X-rays is time-consuming and prone to subjective errors. Automating this process enhances clinical efficiency and consistency, especially in large-scale dental research or automated diagnostic systems.

Key challenges include:
- Variability in tooth presentation due to occlusions, overlapping, and imaging conditions.
- Anatomical correctness and adherence to the FDI quadrant-position tooth numbering.
- Ensuring high detection accuracy and minimal false positives or missed detections.

---

## Dataset Description

The dataset consists of approximately 500 high-resolution dental panoramic radiograph images paired with YOLO formatted annotations describing bounding boxes for each tooth.

- Images reside under `ToothNumber_TaskDataset/images/`.
- Annotations (label files) are under `ToothNumber_TaskDataset/labels/`.
- Each label `.txt` file corresponds exactly by basename to an image file and contains bounding box coordinates normalized to image size.
- The full dataset is automatically cleaned, validated, and split into training (80%), validation (10%), and testing (10%) subsets by the project pipeline.

---

## FDI Tooth Numbering System Explanation

The FDI (Fédération Dentaire Internationale) system uniquely identifies each tooth by a two-digit code:

- The **first digit** (1–4) denotes the quadrant:
  - 1: Upper Right
  - 2: Upper Left
  - 3: Lower Left
  - 4: Lower Right

- The **second digit** (1–8) indicates the tooth position within that quadrant:
  - 1: Central Incisor
  - 2: Lateral Incisor
  - 3: Canine
  - 4: First Premolar
  - 5: Second Premolar
  - 6: First Molar
  - 7: Second Molar
  - 8: Third Molar

This coding provides a standardized, internationally recognized reference for tooth identification in clinical practice and research.

Example: Tooth "11" refers to Upper Right Central Incisor, "36" is Lower Left First Molar.

---

## Project Workflow

![Workflow Diagram](https://github.com/Srinu363/Dental/blob/master/runs/tooth_detection/generated-image.png)
*Figure 2: Project Workflow from Data Preparation to Modeling and Evaluation*

1. **Dataset validation and cleansing:**
   - Remove images with missing or malformed labels.
   - Correct label files to ensure strict YOLO format compliance.
2. **Dataset splitting:**
   - Stratified 80/10/10 split into train/val/test subsets.
3. **Training YOLOv8 model:**
   - Use pretrained YOLOv8s weights as a baseline.
   - Train on train set, monitor on validation set.
4. **Model evaluation:**
   - Compute class-wise precision, recall, mAP@50, and mAP@50-95.
   - Generate confusion matrices to assess misclassifications.
5. **Visualization:**
   - Display sample predictions with bounding boxes and FDI labels.
6. **Post-processing (optional):**
   - Anatomically coherent corrections based on tooth position and spacing.
7. **Results export and documentation:**
   - Save model weights, metrics, sample images, and training logs.

---

## Setup Instructions

### Environment Requirements

- Python 3.8+
- PyTorch compatible with your CUDA version (if GPU available)
- Essential Python libraries (`opencv-python`, `matplotlib`, `pyyaml`, `ultralytics`)

### Installation Commands



---

## Running the Project

To execute the entire pipeline from dataset preparation through training and evaluation:


- Make sure your dataset folder `ToothNumber_TaskDataset` with `images/` and `labels/` is in the same directory.
- Trained model weights and logs will be saved under `runs/tooth_detection/`.
- Dataset splits will appear in `datasets_split/`.

---

## Training Details and Parameters

- Base model: YOLOv8s pretrained weights (`yolov8s.pt`)
- Input image size: 640×640 pixels
- Batch size: 16 (adjustable based on hardware)
- Epochs: 100 (recommended for balance between accuracy and training time)
- Device auto-detected (GPU if available, else CPU)
- Optimizer and augmentation: defaults from Ultralytics YOLOv8 framework

---

## Evaluation and Metrics

The model evaluation includes the following metrics computed on the validation set after each epoch:

- **Precision:** How many detected teeth were correct.
- **Recall:** How many actual teeth were detected.
- **mAP@0.5:** Mean average precision at 50% IoU threshold.
- **mAP@0.5–0.95:** Mean average precision averaged over multiple IoU thresholds.

Confusion matrices visualize the per-class prediction performance highlighting common misclassification trade-offs between similar tooth classes.

---

## Results and Visualizations

![Sample Prediction](https://github.com/Srinu363/Dental/blob/master/runs/tooth_detection/train_batch1.jpg)
*Figure 3: Model prediction example with tooth bounding boxes and FDI labels.*



---

## Post-Processing Logic

To improve anatomical correctness, post-processing modules can:

- Separate upper and lower dental arches via Y-axis clustering.
- Divide left and right quadrants using X-midline detection.
- Sort teeth sequentially in each quadrant following FDI numbering.
- Handle missing or overlapping teeth by adjusting bounding box order and skipping numbering.

---

## Project Structure

.
├── ToothNumber_TaskDataset/
│   ├── images/
│   └── labels/
├── datasets_split/
│   ├── train/
│   ├── val/
│   └── test/
├── runs/
│   └── tooth_detection/
│ ├── weights/
│ ├── results.csv
│ └── ...
├── dental_yolo_training.py
├── README.md
└── requirements.txt


