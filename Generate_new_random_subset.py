import json
import random
import os
import shutil

train_json_path = "json_files/data_list_train.json"
val_json_path = "json_files/data_list_val.json"

output_train_json = "json_files/train_samples.json"
output_val_json = "json_files/val_samples.json"

source_train_image_dir = "LV-MHP-v2/train/images"
source_train_mask_dir = "LV-MHP-v2/train/parsing_annos"

source_val_image_dir = "LV-MHP-v2/val/images"
source_val_mask_dir = "LV-MHP-v2/val/parsing_annos"

train_image_output_dir = "dataset/train/images"
train_mask_output_dir = "dataset/train/masks"

val_image_output_dir = "dataset/val/images"
val_mask_output_dir = "dataset/val/masks"

train_samples = 500
val_samples = 120

os.makedirs(train_image_output_dir, exist_ok=True)
os.makedirs(train_mask_output_dir, exist_ok=True)
os.makedirs(val_image_output_dir, exist_ok=True)
os.makedirs(val_mask_output_dir, exist_ok=True)

with open(train_json_path, "r") as f:
    train_data = json.load(f)

with open(val_json_path, "r") as f:
    val_data = json.load(f)

selected_train_images = random.sample(train_data, min(train_samples, len(train_data)))
selected_val_images = random.sample(val_data, min(val_samples, len(val_data)))


for item in selected_train_images:
    image_name = os.path.basename(item["filepath"])
    correct_image_path = os.path.join(source_train_image_dir, image_name)
    item["filepath"] = os.path.join(train_image_output_dir, image_name)

    if os.path.exists(correct_image_path):
        shutil.copy(correct_image_path, os.path.join(train_image_output_dir, image_name))
    else:
        print(f"Image not found : {correct_image_path}")

    for bbox in item["bboxes"]:
        mask_name = os.path.basename(bbox["ann_path"])
        correct_mask_path = os.path.join(source_train_mask_dir, mask_name)
        bbox["ann_path"] = os.path.join(train_mask_output_dir, mask_name)

        if os.path.exists(correct_mask_path):
            shutil.copy(correct_mask_path, os.path.join(train_mask_output_dir, mask_name))
        else:
            print(f"Mask not found : {correct_mask_path}")

for item in selected_val_images:
    image_name = os.path.basename(item["filepath"])
    correct_image_path = os.path.join(source_val_image_dir, image_name)
    item["filepath"] = os.path.join(val_image_output_dir, image_name)

    if os.path.exists(correct_image_path):
        shutil.copy(correct_image_path, os.path.join(val_image_output_dir, image_name))
    else:
        print(f"Image not found : {correct_image_path}")

    for bbox in item["bboxes"]:
        mask_name = os.path.basename(bbox["ann_path"])
        correct_mask_path = os.path.join(source_val_mask_dir, mask_name)
        bbox["ann_path"] = os.path.join(val_mask_output_dir, mask_name)

        if os.path.exists(correct_mask_path):
            shutil.copy(correct_mask_path, os.path.join(val_mask_output_dir, mask_name))
        else:
            print(f"Mask not found : {correct_mask_path}")

with open(output_train_json, "w") as f:
    json.dump(selected_train_images, f, indent=4)

with open(output_val_json, "w") as f:
    json.dump(selected_val_images, f, indent=4)

print(f"Dataset created successfully.")
print(f"JSON file updated.")
