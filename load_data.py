import json
import cv2
import numpy as np
import os
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from merge_json import merge_json

train_json_path = "json_files/train_samples_with_gender.json"
val_json_path = "json_files/val_samples_with_gender.json"

output_train_json = "json_files/LV_MHP_V2_train.json"
output_val_json = "json_files/LV_MHP_V2_val.json"

MIN_INTERNAL_AREA = 500

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Mask not found: {mask_path}")
        return None

    _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(contour)
                if area >= MIN_INTERNAL_AREA:
                    min_dist = float("inf")
                    best_pair = None
                    for j, outer_contour in enumerate(contours):
                        if hierarchy[0][j][3] == -1:
                            for inner_point in contour:
                                for outer_point in outer_contour:
                                    dist = np.linalg.norm(inner_point - outer_point)
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_pair = (tuple(inner_point[0]), tuple(outer_point[0]))
                    if best_pair:
                        cv2.line(mask_binary, best_pair[0], best_pair[1], 0, thickness=3)

    return mask_binary


def get_detectron2_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    dataset_dicts = []

    for idx, item in enumerate(data):
        record = {
            "file_name": item["filepath"],
            "image_id": idx,
            "height": item["height"],
            "width": item["width"],
            "annotations": []
        }

        for ann in item["bboxes"]:
            mask_path = ann["ann_path"]
            bbox = [ann["x1"], ann["y1"], ann["x2"], ann["y2"]]
            category_id = 0 if ann["class"] == "male" else 1

            mask_binary = process_mask(mask_path)
            if mask_binary is None:
                continue

            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = [contour.flatten().tolist() for contour in contours if len(contour) > 4]

            if not segmentation:
                continue

            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": segmentation,
                "category_id": category_id
            }
            record["annotations"].append(obj)

        dataset_dicts.append(record)

    return dataset_dicts


def register_dataset():
    datasets = {
        "LV_MHP_V2_train": (train_json_path, output_train_json),
        "LV_MHP_V2_val": (val_json_path, output_val_json)
    }

    for dataset_name, (input_json, output_json) in datasets.items():
        if os.path.exists(output_json):
            print(f"{dataset_name} is already processed and registered.")
        else:
            print(f"Processing {dataset_name} ...")
            dataset_data = get_detectron2_dataset(input_json)

            with open(output_json, "w") as f:
                json.dump(dataset_data, f, indent=4)
            print(f"The dataset has been registered: {dataset_name}")

        DatasetCatalog.register(dataset_name, lambda path=output_json: json.load(open(path, "r")))
        MetadataCatalog.get(dataset_name).set(thing_classes=["male", "female"])
        
register_dataset()
