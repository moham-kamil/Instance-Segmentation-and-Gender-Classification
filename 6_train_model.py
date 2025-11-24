from load_data import *
import torch
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
import os

train_dataset_name = "LV_MHP_V2_train"
val_dataset_name = "LV_MHP_V2_val"

if train_dataset_name and val_dataset_name not in DatasetCatalog.list():
    register_dataset()

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = (train_dataset_name,)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.OUTPUT_DIR = output_dir

cfg.dump(stream=open(f"{output_dir}/config.yaml", "w"))

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

trainer.train()

