import os
import json
import shutil

# Define dataset root path
LV_MHP_V2_PATH = "/media/mk/Volume/CV Task/LV-MHP-v2"  # Change this to the actual dataset path

# Define source paths inside LV-MHP-v2
train_images_src = os.path.join(LV_MHP_V2_PATH, "train", "images")
train_masks_src = os.path.join(LV_MHP_V2_PATH, "train", "parsing_annos")

val_images_src = os.path.join(LV_MHP_V2_PATH, "val", "images")
val_masks_src = os.path.join(LV_MHP_V2_PATH, "val", "parsing_annos")

# Define JSON paths
train_json_path = "json_files/train_samples.json"
val_json_path = "json_files/val_samples.json"

# Define destination directories
output_dir = "dataset"
train_output_dir = os.path.join(output_dir, "train")
val_output_dir = os.path.join(output_dir, "val")

# Create output directories
os.makedirs(os.path.join(train_output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_output_dir, "masks"), exist_ok=True)
os.makedirs(os.path.join(val_output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(val_output_dir, "masks"), exist_ok=True)

def copy_samples(json_path, images_src, masks_src, output_dir):
    with open(json_path, "r") as f:
        data = json.load(f)

    for item in data:
        # Copy image
        image_name = os.path.basename(item["filepath"])
        image_src = os.path.join(images_src, image_name)
        image_dst = os.path.join(output_dir, "images", image_name)
        if os.path.exists(image_src):
            shutil.copy(image_src, image_dst)

        # Copy masks
        for ann in item["bboxes"]:
            mask_name = os.path.basename(ann["ann_path"])
            mask_src = os.path.join(masks_src, mask_name)
            mask_dst = os.path.join(output_dir, "masks", mask_name)
            if os.path.exists(mask_src):
                shutil.copy(mask_src, mask_dst)

# Copy train and val samples
copy_samples(train_json_path, train_images_src, train_masks_src, train_output_dir)
copy_samples(val_json_path, val_images_src, val_masks_src, val_output_dir)

print("Samples copied successfully.")
