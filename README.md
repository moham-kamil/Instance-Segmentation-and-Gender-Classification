# README - Running the Model

## Requirements

Ensure you have the following installed:

- Python 3.8+
- OpenCV
- PyTorch
- Detectron2
- Matplotlib
- Numpy

## Setup

1. Clone or download the project.
2. Install required dependencies.
3. Copy the trained model sent on Telegram [model_0009999.pth] to `./output/`.
5. To run the code faster, copy the prepared JSON files sent on Telegram [LV_MHP_V2_val] and [LV_MHP_V2_train] to `./json_files`.

## Running the Code

### Data Visualization

To display images with overlaid bounding boxes and parsing annotations, Change the path [data_root] to the actual dataset path , then run:

```bash
python 0_visualize_mhp_data.py
```

### Generate Subset from LV-MHP-v2

Change the path [LV_MHP_V2_PATH] to the actual dataset path , then run:

```bash
python 1_Generate_subset.py
```

### Mask Generation

To remove masks from face and hands, run:

```bash
python 3_Mask_generation.py
```

### Visualize Masks

To visualize the masks on the sample of images , run:

```bash
python 4_visualize_mask.py
```

This will display sample images with their annotated masks.


### Visualize Training Data

To visualize the dataset before training, run:

```bash
python 5_visualize_train_data.py
```

This will display sample images with their annotated masks and bounding boxes.

### Train the Model

Run:

```bash
python 6_train_model.py
```

### Test the Model & Visualize Predictions

Run:

```bash
python 7_test_model.py
```

This will display images with detected objects, masks, and bounding boxes.

### Evaluate Model Performance

To evaluate the trained model on the validation set, run:

```bash
python 8_evaluate_model.py
```

## Notes

- **Male** objects will be highlighted in **green**.
- **Female** objects will be highlighted in **red**.


