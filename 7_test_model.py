from load_data import *
import torch
import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt
import random

train_dataset_name = "LV_MHP_V2_train"
val_dataset_name = "LV_MHP_V2_val"

if train_dataset_name not in DatasetCatalog.list() or val_dataset_name not in DatasetCatalog.list():
    register_dataset()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./output/model_0009999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.DATASETS.TEST = (val_dataset_name,)

predictor = DefaultPredictor(cfg)
dataset_dicts = DatasetCatalog.get(val_dataset_name)

sample = random.sample(dataset_dicts, 5)

for item in sample:
    image_path = item["file_name"]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    metadata = MetadataCatalog.get(train_dataset_name)
    visualizer = Visualizer(image, metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    vis_output = visualizer.draw_instance_predictions(instances)

    masks = instances.pred_masks.to("cpu").numpy()
    boxes = instances.pred_boxes.tensor.numpy()
    labels = instances.pred_classes.numpy()
    class_names = metadata.get("thing_classes", ["Male", "Female"])

    for i, mask in enumerate(masks):
        label = labels[i]
        color = (0, 255, 0) if label == 0 else (255, 0, 0)

        image[mask] = color

        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = class_names[label]
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
