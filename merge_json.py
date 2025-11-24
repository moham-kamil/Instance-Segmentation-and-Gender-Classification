import json
import os

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def update_gender_labels(lv_mhp_data, coco_data):
    gender_dict = {}

    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        x, y, w, h = annotation["bbox"]
        bbox = [x, y, x + w, y + h]
        gender_label = "male" if category_id == 1 else "female"

        if image_id not in gender_dict:
            gender_dict[image_id] = []

        gender_dict[image_id].append({"bbox": bbox, "gender": gender_label})

    for entry in lv_mhp_data:
        file_name = os.path.basename(entry["filepath"])
        matching_image = next((img for img in coco_data["images"] if img["file_name"] == file_name), None)
        if not matching_image:
            continue

        image_id = matching_image["id"]
        if image_id in gender_dict:
            annotations = gender_dict[image_id]

            for bbox_entry in entry["bboxes"]:
                x1, y1, x2, y2 = bbox_entry["x1"], bbox_entry["y1"], bbox_entry["x2"], bbox_entry["y2"]
                best_iou = 0.0
                best_match = None

                for bbox_info in annotations:
                    iou = calculate_iou([x1, y1, x2, y2], bbox_info["bbox"])
                    if iou > best_iou and iou > 0.5:
                        best_iou = iou
                        best_match = bbox_info

                if best_match:
                    bbox_entry["class"] = best_match["gender"]

def merge_json():
    coco_train_json_path = "json_files/gender_labels_train.json"
    coco_val_json_path = "json_files/gender_labels_val.json"

    with open(coco_train_json_path, "r") as f:
        coco_train_data = json.load(f)

    with open(coco_val_json_path, "r") as f:
        coco_val_data = json.load(f)

    lv_mhp_train_json_path = "json_files/train_samples.json"
    lv_mhp_val_json_path = "json_files/val_samples.json"

    with open(lv_mhp_train_json_path, "r") as f:
        lv_mhp_train_data = json.load(f)

    with open(lv_mhp_val_json_path, "r") as f:
        lv_mhp_val_data = json.load(f)

    update_gender_labels(lv_mhp_train_data, coco_train_data)
    update_gender_labels(lv_mhp_val_data, coco_val_data)

    updated_lv_mhp_train_json_path = "json_files/train_samples_with_gender.json"
    updated_lv_mhp_val_json_path = "json_files/val_samples_with_gender.json"

    with open(updated_lv_mhp_train_json_path, "w") as f:
        json.dump(lv_mhp_train_data, f, indent=4)

    with open(updated_lv_mhp_val_json_path, "w") as f:
        json.dump(lv_mhp_val_data, f, indent=4)

    print("JSON files have been merged successfully.")

merge_json()
