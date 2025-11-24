import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def overlay_masks(image, masks, alpha=1):
    overlaid = image.copy()

    for mask in masks:
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8).tolist()[0]
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 0] = color[0]
        mask_colored[:, :, 1] = color[1]
        mask_colored[:, :, 2] = color[2]

        mask_overlay = np.where(mask > 0, mask_colored, overlaid)

        overlaid = cv2.addWeighted(overlaid, 1 - alpha, mask_overlay, alpha, 0)

    return overlaid


def visualize_masks(images, masks_list, titles=None):

    n = len(images)
    plt.figure(figsize=(15, 5))

    for i in range(n):
        overlaid = overlay_masks(images[i], masks_list[i])

        plt.subplot(1, n, i + 1)
        plt.imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        if titles:
            plt.title(titles[i])

    plt.show()


json_path = "json_files/train_samples.json"

with open(json_path, "r") as f:
    data = json.load(f)

sample_data = random.sample(data, min(5, len(data)))

for item in sample_data:
    image_path = item["filepath"]
    masks_paths = [ann["ann_path"] for ann in item["bboxes"]]

    image = cv2.imread(image_path)

    masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in masks_paths]

    if image is not None and all(mask is not None for mask in masks):
        visualize_masks([image], [masks], [f"Masks for {os.path.basename(image_path)}"])
    else:
        print(f"Some files were not found for {image_path}")
