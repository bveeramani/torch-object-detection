import collections
from functools import partial
from typing import Callable, Dict, Literal, Tuple

import numpy as np
import ray
import torch
from PIL import Image
from ray import train
from ray.air import Checkpoint, session
from ray.air.config import ScalingConfig
from ray.air.util.tensor_extensions.pandas import _create_possibly_ragged_ndarray
from ray.train.batch_predictor import BatchPredictor
from ray.train.torch import TorchPredictor, TorchTrainer
from torchvision import datasets, models, transforms

name_to_label = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}


def wrangle_data(batch: Tuple[Image.Image, Dict]) -> Dict[str, np.ndarray]:
    output = collections.defaultdict(list)
    for image, target in batch:
        output["image"].append(np.array(image))

        boxes = []
        labels = []
        for object in target["annotation"]["object"]:
            x1 = int(object["bndbox"]["xmin"])
            y1 = int(object["bndbox"]["ymin"])
            x2 = int(object["bndbox"]["xmax"])
            y2 = int(object["bndbox"]["ymax"])
            boxes.append((x1, y1, x2, y2))

            label = name_to_label[object["name"]]
            labels.append(label)

        output["boxes"].append(np.array(boxes))
        output["labels"].append(np.array(labels))

    return {
        key: _create_possibly_ragged_ndarray(value) for key, value in output.items()
    }


def preprocess(
    batch: Dict[str, np.ndarray], transform: Callable[[np.ndarray], torch.Tensor]
) -> Dict[str, np.ndarray]:
    # TODO: Use `TorchvisionPreprocessor`
    batch["image"] = _create_possibly_ragged_ndarray(
        [transform(image).numpy() for image in batch["image"]]
    )
    return batch


def get_dataset(
    root: str,
    *,
    split: Literal["train", "val"],
    transform: Callable[[np.ndarray], torch.Tensor],
):
    dataset = datasets.VOCDetection(root, download=True, image_set=split)
    return (
        ray.data.from_torch(dataset)
        .map_batches(wrangle_data)
        .map_batches(partial(preprocess, transform=transform), batch_format="numpy")
    )


train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ConvertImageDtype(torch.float),
    ]
)
train_dataset = get_dataset("data", split="train", transform=train_transform)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ]
)
val_dataset = get_dataset("data", split="val", transform=val_transform)


def train_one_epoch(*, model, optimizer, batch_size, epoch):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=1000
        )

    device = ray.train.torch.get_device()
    train_dataset_shard = session.get_dataset_shard("train")
    batches = train_dataset_shard.iter_batches(batch_size=batch_size)
    for batch in batches:
        inputs = [torch.as_tensor(image).to(device) for image in batch["image"]]
        targets = [
            {
                "boxes": torch.as_tensor(boxes).to(device),
                "labels": torch.as_tensor(labels).to(device),
                "masks": torch.as_tensor(masks).to(device),
            }
            for boxes, labels, masks in zip(
                batch["boxes"], batch["labels"], batch["masks"]
            )
        ]
        loss_dict = model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        session.report(
            {
                "losses": losses.item(),
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                **{key: value.item() for key, value in loss_dict.items()},
            }
        )


def train_loop_per_worker(config):
    model = models.detection.fasterrcnn_resnet50_fpn()
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
    start_epoch = 0

    checkpoint = session.get_checkpoint()
    if checkpoint is not None:
        checkpoint_data = checkpoint.to_dict()
        model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        lr_scheduler.load_state_dict(checkpoint_data["lr_scheduler"])
        start_epoch = checkpoint_data["epoch"]

    for epoch in range(start_epoch, config["epochs"]):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            batch_size=config["batch_size"],
            epoch=epoch,
        )
        lr_scheduler.step()
        checkpoint = Checkpoint.from_dict(
            {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
            }
        )
        session.report({}, checkpoint=checkpoint)


trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        "batch_size": 2,
        "lr": 0.02,
        # "epochs": 26,
        "epochs": 1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lr_steps": [16, 22],
        "lr_gamma": 0.1,
    },
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
    datasets={"train": train_dataset},
)
results = trainer.fit()


model = models.detection.fasterrcnn_resnet50_fpn()


class CustomTorchPredictor(TorchPredictor):
    def _predict_numpy(
        self, data: np.ndarray, dtype: torch.dtype
    ) -> Dict[str, np.ndarray]:
        device = torch.device("cuda")
        inputs = [
            torch.as_tensor(image, dtype=dtype, device=device)
            for image in data["image"]
        ]
        outputs = self.call_model(inputs)

        # Convert outputs to NumPy batch format
        predictions = collections.defaultdict(list)
        for output in outputs:
            for key, value in output.items():
                predictions[key].append(value.cpu().detach().numpy())
        for key, value in predictions.items():
            predictions[key] = _create_possibly_ragged_ndarray(value)

        # Rename keys to prevent collisions with ground truth
        predictions = {"pred_" + key: value for key, value in predictions.items()}

        return predictions


predictor = BatchPredictor(results.checkpoint, CustomTorchPredictor, model=model)
predictions = predictor.predict(
    val_dataset,
    feature_columns=["image"],
    keep_columns=["boxes", "labels"],
    batch_size=4,
)


from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision()
for row in predictions.iter_rows():
    preds = [
        {
            "boxes": torch.as_tensor(row["pred_boxes"]),
            "scores": torch.as_tensor(row["pred_scores"]),
            "labels": torch.as_tensor(row["pred_labels"]),
        }
    ]
    target = [
        {
            "boxes": torch.as_tensor(row["boxes"]),
            "labels": torch.as_tensor(row["labels"]),
        }
    ]
    metric.update(preds, target)
print(metric.compute())
