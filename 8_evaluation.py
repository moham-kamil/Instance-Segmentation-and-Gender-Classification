from load_data import *
import torch
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
from detectron2.data import DatasetCatalog, MetadataCatalog


train_dataset_name = "LV_MHP_V2_train"
val_dataset_name = "LV_MHP_V2_val"

if train_dataset_name and val_dataset_name not in DatasetCatalog.list():
    register_dataset()


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (train_dataset_name,)
cfg.DATASETS.TEST = (val_dataset_name,)
cfg.MODEL.WEIGHTS = "./output/model_0009999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)

dataset_dicts = DatasetCatalog.get(val_dataset_name)

evaluator = COCOEvaluator(val_dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, val_dataset_name)

print("The data is ready, and the evaluation is now running...")
metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
print("Evaluation Successful.")
print(metrics)
