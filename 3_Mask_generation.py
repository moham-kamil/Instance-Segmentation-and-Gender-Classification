import json
import cv2
import numpy as np
import os

train_json_path = "json_files/train_samples.json"
val_json_path = "json_files/val_samples.json"

train_output_mask_dir = "dataset/train/masks"
val_output_mask_dir = "dataset/val/masks"

os.makedirs(train_output_mask_dir, exist_ok=True)
os.makedirs(val_output_mask_dir, exist_ok=True)


with open(train_json_path, "r") as f:
    train_data = json.load(f)

with open(val_json_path, "r") as f:
    val_data = json.load(f)

EXCLUDED_PARTS = [0, 2]

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Mask not found : {mask_path}")
        return None

    for part in EXCLUDED_PARTS:
        mask[mask == part] = 0

    mask[mask > 0] = 255

    return mask

for item in train_data:
    for ann in item["bboxes"]:
        mask_path = ann["ann_path"]
        mask_name = os.path.basename(mask_path)
        train_output_mask_path = os.path.join(train_output_mask_dir, mask_name)

        mask = process_mask(mask_path)
        if mask is not None:
            cv2.imwrite(train_output_mask_path, mask)

for item in val_data:
    for ann in item["bboxes"]:
        mask_path = ann["ann_path"]
        mask_name = os.path.basename(mask_path)
        val_output_mask_path = os.path.join(val_output_mask_dir, mask_name)

        mask = process_mask(mask_path)
        if mask is not None:
            cv2.imwrite(val_output_mask_path, mask)

print("Masks generated successfully. Face and hands removed from masks.")
