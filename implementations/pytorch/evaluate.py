"""测试保存的模型。"""

import argparse
from pathlib import Path

from torch import nn

from data import build_test_loader
from engine import evaluate
from model import CatDogCNN
from utils import load_checkpoint, select_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 PyTorch 猫狗分类器")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "test",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pytorch" / "best_model_pytorch.pt",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    loader, class_to_idx = build_test_loader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = load_checkpoint(args.model_path)
    if checkpoint.get("class_to_idx") != class_to_idx:
        raise ValueError("模型类别映射与测试集不一致")

    model = CatDogCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss, accuracy, matrix = evaluate(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        with_confusion_matrix=True,
    )

    print(f"设备：{device}")
    print(f"测试样本：{len(loader.dataset)}")
    print(f"测试损失：{loss:.4f}")
    print(f"测试准确率：{accuracy * 100:.2f}%")
    print("\n混淆矩阵：")
    print("          预测猫  预测狗")
    print(f"真实猫：{matrix[0, 0].item():6d} {matrix[0, 1].item():6d}")
    print(f"真实狗：{matrix[1, 0].item():6d} {matrix[1, 1].item():6d}")

    for label, name in ((0, "猫"), (1, "狗")):
        class_total = matrix[label].sum().item()
        class_accuracy = matrix[label, label].item() / class_total
        print(f"{name}识别正确率：{class_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
