"""训练与评估循环。"""

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_number, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if batch_number % 20 == 0 or batch_number == len(loader):
            print(
                f"\r训练批次：{batch_number}/{len(loader)}",
                end="",
                flush=True,
            )

    print()
    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    with_confusion_matrix: bool = False,
) -> tuple[float, float, torch.Tensor | None]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion_matrix = (
        torch.zeros((2, 2), dtype=torch.int64)
        if with_confusion_matrix
        else None
    )

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        if confusion_matrix is not None:
            flat_indices = labels.cpu() * 2 + predictions.cpu()
            confusion_matrix += torch.bincount(
                flat_indices, minlength=4
            ).reshape(2, 2)

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        confusion_matrix,
    )

