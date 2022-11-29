from typing import Any
import json
import os
import warnings

import torch
import numpy as np
import torchvision

import ray
from ray.air.util.tensor_extensions.pandas import _create_possibly_ragged_ndarray
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


def convert_path_to_filename(batch: dict[str, Any]) -> dict[str, Any]:
    batch["filename"] = np.array([os.path.basename(path) for path in batch["path"]])
    del batch["path"]
    return batch


def add_boxes(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    batch["boxes"] = []

    for image, filename in zip(batch["image"], batch["filename"]):
        annotations = filename_to_annotations[filename]
        boxes = [annotation["bbox"] for annotation in annotations]
        boxes = np.stack(boxes)

        # (X, Y, W, H) -> (X1, Y1, X2, Y2)
        boxes[:, 2:] += boxes[:, :2]

        height, width = image.shape[0:2]
        boxes[:, 0::2].clip(min=0, max=width)
        boxes[:, 1::2].clip(min=0, max=height)

        batch["boxes"].append(boxes)

    batch["boxes"] = np.array(batch["boxes"])
    return batch


def add_labels(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    batch["labels"] = []

    for filename in batch["filename"]:
        annotations = filename_to_annotations[filename]
        labels = np.array([annotation["category_id"] for annotation in annotations])
        batch["labels"].append(labels)

    batch["labels"] = np.array(batch["labels"])
    return batch


def add_masks(batch):
    batch["masks"] = []

    for image, filename in zip(batch["image"], batch["filename"]):
        annotations = filename_to_annotations[filename]
        segmentations = [annotation["segmentation"] for annotation in annotations]
        height, width = image.shape[0:2]
        masks = convert_coco_poly_to_mask(segmentations, height, width)
        batch["masks"].append(masks)

    batch["masks"] = np.array(batch["masks"])

    return batch


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.array(mask, dtype=np.uint8)
        mask = mask.any(axis=2)
        masks.append(mask)
    if masks:
        try:
            masks = np.stack(masks, axis=0)
        except ValueError as e:
            for mask in masks:
                if mask.shape != masks[0].shape:
                    print(mask.shape, masks[0].shape)
            masks = np.zeros((0, height, width), dtype=np.uint8)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)
    return masks


def preprocess(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    transform = torchvision.transforms.ToTensor()
    # FIXME: (_map_block_split pid=36622) ValueError: could not broadcast input array from shape (3,426,640) into shape (3,)
    # batch["image"] = np.array([transform(image).numpy() for image in batch["image"]])
    batch["image"] = _create_possibly_ragged_ndarray(
        [transform(image).numpy() for image in batch["image"]]
    )
    return batch


root = "/Users/balaji/Datasets/COCO/"
coco = COCO(os.path.join(root, "annotations", "instances_val2017.json"))

filename_to_annotations = {}
for image_id in coco.getImgIds():
    images: list[dict] = coco.loadImgs(image_id)
    assert len(images) == 1
    filename = images[0]["file_name"]

    annotation_ids = coco.getAnnIds(imgIds=image_id)
    if len(annotation_ids) == 0:
        warnings.warn(f"Image with ID {image_id} doesn't have any annotations.")
        continue

    annotations = coco.loadAnns(annotation_ids)
    assert len(annotations) > 0
    filename_to_annotations[filename] = annotations


ds = (
    ray.data.read_images(os.path.join(root, "val2017"), mode="RGB", include_paths=True)
    .limit(10)
    .map_batches(convert_path_to_filename, batch_format="numpy")
    .filter(lambda record: record["filename"] in filename_to_annotations)
    .map_batches(add_boxes, batch_format="numpy")
    .map_batches(add_labels, batch_format="numpy")
    .map_batches(add_masks, batch_format="numpy")
    .map_batches(preprocess, batch_format="numpy")
)
print(ds)


batch = next(iter(ds.iter_batches(batch_size=2, batch_format="numpy")))
inputs = [torch.as_tensor(image) for image in batch["image"]]
targets = {
    "boxes": list(batch["boxes"]),
    "labels": list(batch["labels"]),
    "masks": list(batch["masks"]),
}
targets = [
    {
        "boxes": torch.as_tensor(boxes),
        "labels": torch.as_tensor(labels),
        "masks": torch.as_tensor(masks),
    }
    for boxes, labels, masks in zip(batch["boxes"], batch["labels"], batch["masks"])
]
model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
)
