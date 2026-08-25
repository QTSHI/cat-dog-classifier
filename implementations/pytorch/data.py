"""数据读取与增强。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


IMAGE_SIZE = 128
EXPECTED_CLASSES = {"cats": 0, "dogs": 1}
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def training_transform():
    """训练图片的处理方式。"""

    return v2.Compose(
        [
            v2.RandomResizedCrop(
                (IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def evaluation_transform():
    """验证和测试图片的处理方式。"""

    return v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def _check_dataset(dataset: ImageFolder, root: Path) -> None:
    if dataset.class_to_idx != EXPECTED_CLASSES:
        raise ValueError(
            f"{root} 的类别目录应为 cats 和 dogs，"
            f"实际映射为 {dataset.class_to_idx}"
        )


def stratified_split_indices(
    targets: list[int],
    validation_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """分别划分猫和狗，避免比例变化。"""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio 必须在 0 和 1 之间")

    generator = torch.Generator().manual_seed(seed)
    target_tensor = torch.as_tensor(targets)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for label in sorted(set(targets)):
        class_indices = torch.where(target_tensor == label)[0]
        order = torch.randperm(len(class_indices), generator=generator)
        class_indices = class_indices[order]
        validation_count = int(len(class_indices) * validation_ratio)

        validation_indices.extend(
            class_indices[:validation_count].tolist()
        )
        train_indices.extend(class_indices[validation_count:].tolist())

    train_order = torch.randperm(len(train_indices), generator=generator)
    validation_order = torch.randperm(
        len(validation_indices), generator=generator
    )

    train_indices = [train_indices[index] for index in train_order.tolist()]
    validation_indices = [
        validation_indices[index] for index in validation_order.tolist()
    ]
    return train_indices, validation_indices


def build_train_validation_loaders(
    data_dir: Path,
    batch_size: int,
    validation_ratio: float,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"找不到训练数据目录：{data_dir}")

    index_dataset = ImageFolder(data_dir)
    _check_dataset(index_dataset, data_dir)
    train_indices, validation_indices = stratified_split_indices(
        index_dataset.targets,
        validation_ratio,
        seed,
    )

    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]
    if max_validation_samples is not None:
        validation_indices = validation_indices[:max_validation_samples]

    train_dataset = Subset(
        ImageFolder(data_dir, transform=training_transform()),
        train_indices,
    )
    validation_dataset = Subset(
        ImageFolder(data_dir, transform=evaluation_transform()),
        validation_indices,
    )

    generator = torch.Generator().manual_seed(seed)
    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_options,
    )
    return train_loader, validation_loader, index_dataset.class_to_idx


def build_test_loader(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, dict[str, int]]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"找不到测试数据目录：{data_dir}")

    dataset = ImageFolder(data_dir, transform=evaluation_transform())
    _check_dataset(dataset, data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return loader, dataset.class_to_idx
