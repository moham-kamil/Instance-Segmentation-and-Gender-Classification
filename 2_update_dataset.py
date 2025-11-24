import json
import os

train_json_path = "json_files/train_samples.json"
val_json_path = "json_files/val_samples.json"

train_image_dir = "dataset/train/images"
train_mask_dir = "dataset/train/masks"

val_image_dir = "dataset/val/images"
val_mask_dir = "dataset/val/masks"

with open(train_json_path, "r") as f:
    train_data = json.load(f)

with open(val_json_path, "r") as f:
    val_data = json.load(f)

def update_json(data, image_dir, mask_dir):
    updated_data = []
    changes_detected = False

    existing_masks = set(os.listdir(mask_dir))

    for item in data:
        image_name = os.path.basename(item["filepath"])
        image_path = os.path.join(image_dir, image_name)

        if os.path.exists(image_path):
            item["filepath"] = image_path
        else:
            print(f"Image not found : {image_name}")
            changes_detected = True
            continue

        new_bboxes = []
        for bbox in item["bboxes"]:
            mask_name = os.path.basename(bbox["ann_path"])
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                bbox["ann_path"] = mask_path
                new_bboxes.append(bbox)
            else:
                print(f"Mask not found : {mask_name}")
                changes_detected = True


        if new_bboxes:
            item["bboxes"] = new_bboxes
            updated_data.append(item)

    return updated_data, changes_detected

updated_train_data, train_changes = update_json(train_data, train_image_dir, train_mask_dir)
updated_val_data, val_changes = update_json(val_data, val_image_dir, val_mask_dir)

def remove_unused_masks(mask_dir, json_data):
    used_masks = {os.path.basename(bbox["ann_path"]) for item in json_data for bbox in item["bboxes"]}
    all_masks = set(os.listdir(mask_dir))

    unused_masks = all_masks - used_masks

    if unused_masks:
        for mask in unused_masks:
            mask_path = os.path.join(mask_dir, mask)
            os.remove(mask_path)

remove_unused_masks(train_mask_dir, updated_train_data)
remove_unused_masks(val_mask_dir, updated_val_data)

if train_changes:
    with open(train_json_path, "w") as f:
        json.dump(updated_train_data, f, indent=4)
    print(f"Train Dataset updated successfully.")
    print(f"Train JSON file updated.")

if val_changes:
    with open(val_json_path, "w") as f:
        json.dump(updated_val_data, f, indent=4)
    print(f"Val Dataset updated successfully.")
    print(f"Val JSON file updated.")

if not train_changes and not val_changes:
    print("Dataset is up to date.")
