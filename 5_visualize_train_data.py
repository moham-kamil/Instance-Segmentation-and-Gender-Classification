from load_data import *
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

train_dataset_name = "LV_MHP_V2_train"
val_dataset_name = "LV_MHP_V2_val"

if train_dataset_name not in DatasetCatalog.list():
    register_dataset()

dataset_dicts = DatasetCatalog.get(train_dataset_name)
metadata = MetadataCatalog.get(train_dataset_name)

def visualize_sample():
 for n in range(5):
    sample = random.choice(dataset_dicts)
    image_path = sample["file_name"]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for ann in sample["annotations"]:
        x1, y1, x2, y2 = ann["bbox"]
        category_id = ann["category_id"]
        label = "Male" if category_id == 0 else "Female"
        color = (0, 255, 0) if category_id == 0 else (255, 0, 0)

        for segmentation in ann["segmentation"]:
            mask_pts = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(image, [mask_pts], color)

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

visualize_sample()
