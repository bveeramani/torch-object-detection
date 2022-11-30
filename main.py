from typing import Any
import json
import os
import warnings

import torch
import numpy as np
import torchvision

import ray
from ray import train
from ray.air import session
from ray.air.util.tensor_extensions.pandas import _create_possibly_ragged_ndarray
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig


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


val_dataset = (
    ray.data.read_images(os.path.join(root, "val2017"), mode="RGB", include_paths=True)
    .limit(10)
    .map_batches(convert_path_to_filename, batch_format="numpy")
    .filter(lambda record: record["filename"] in filename_to_annotations)
    .map_batches(add_boxes, batch_format="numpy")
    .map_batches(add_labels, batch_format="numpy")
    .map_batches(add_masks, batch_format="numpy")
    .map_batches(preprocess, batch_format="numpy")
)


def train_one_epoch(*, model, optimizer, scaler, batch_size, epoch):
    model.train()

    train_dataset_shard = session.get_dataset_shard("train")
    batches = train_dataset_shard.iter_batches(batch_size=batch_size)
    for batch in batches:
        inputs = [torch.as_tensor(image) for image in batch["image"]]
        targets = [
            {
                "boxes": torch.as_tensor(boxes),
                "labels": torch.as_tensor(labels),
                "masks": torch.as_tensor(masks),
            }
            for boxes, labels, masks in zip(
                batch["boxes"], batch["labels"], batch["masks"]
            )
        ]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(inputs, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        optimizer.step()

        session.report({"losses": losses.item(), "epoch": epoch})


def train_loop_per_worker(config):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    model = train.torch.prepare_model(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler() if config["amp"] else None
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config["lr_steps"], gamma=config["lr_gamma"]
    )
    for epoch in range(0, config["epochs"]):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            batch_size=config["batch_size"],
            epoch=epoch,
        )
        lr_scheduler.step()


# TODO: Add data augmentation
# TODO: Add checkpoint and restore
# TODO: Add reporting
# TODO: Add validation accuracy
# TODO: Add enable reproducibiltiy?
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        "batch_size": 2,
        "lr": 0.02,
        # "epochs": 26,
        "epochs": 1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "amp": False,
        "lr_steps": [16, 22],
        "lr_gamma": 0.1,
    },
    scaling_config=ScalingConfig(num_workers=2),
    datasets={"train": val_dataset},
)
results = trainer.fit()
