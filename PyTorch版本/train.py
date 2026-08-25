"""训练 PyTorch 猫狗分类模型。"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from data import build_train_validation_loaders
from engine import evaluate, train_one_epoch
from model import CatDogCNN
from utils import save_checkpoint, select_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 PyTorch 猫狗分类器")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "数据" / "Preprocessed_Data" / "train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "输出" / "PyTorch",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs 和 batch-size 必须大于 0")

    set_seed(args.seed)
    device = select_device(args.device)
    pin_memory = device.type == "cuda"

    train_loader, validation_loader, class_to_idx = (
        build_train_validation_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
        )
    )

    model = CatDogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_model_path = args.output_dir / "best_model.pt"
    best_accuracy = -1.0
    started_at = time.time()

    print(f"设备：{device}")
    print(f"类别：{class_to_idx}")
    print(
        f"训练集：{len(train_loader.dataset)}，"
        f"验证集：{len(validation_loader.dataset)}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_loss, validation_accuracy, _ = evaluate(
            model, validation_loader, criterion, device
        )
        scheduler.step(validation_loss)

        print(
            f"训练损失：{train_loss:.4f}，"
            f"训练准确率：{train_accuracy * 100:.2f}%"
        )
        print(
            f"验证损失：{validation_loss:.4f}，"
            f"验证准确率：{validation_accuracy * 100:.2f}%"
        )
        print(f"本轮耗时：{time.time() - epoch_started_at:.2f} 秒")

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(
                best_model_path,
                model,
                epoch,
                validation_accuracy,
                class_to_idx,
                vars(args),
            )
            print(f"已保存最佳模型：{best_model_path}")

    print(
        f"\n训练完成，最佳验证准确率：{best_accuracy * 100:.2f}%，"
        f"总耗时：{time.time() - started_at:.2f} 秒"
    )


if __name__ == "__main__":
    main()

