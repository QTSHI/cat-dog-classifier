"""整理模型分错的图片。"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data import EXPECTED_CLASSES, evaluation_transform
from model import CatDogCNN
from utils import load_checkpoint, select_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = {0: "猫", 1: "狗"}


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化猫狗分类错误")
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pytorch",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--examples-per-direction", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    dataset: ImageFolder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[dict]]:
    confusion_matrix = np.zeros((2, 2), dtype=np.int64)
    errors: list[dict] = []
    sample_offset = 0

    for batch_number, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        probabilities = torch.softmax(model(images), dim=1).cpu()
        predictions = probabilities.argmax(dim=1)
        labels = labels.cpu()

        flat_indices = labels * 2 + predictions
        confusion_matrix += (
            torch.bincount(flat_indices, minlength=4)
            .reshape(2, 2)
            .numpy()
        )

        for batch_index in range(labels.size(0)):
            true_label = int(labels[batch_index].item())
            predicted_label = int(predictions[batch_index].item())
            if true_label == predicted_label:
                continue

            image_path = dataset.samples[sample_offset + batch_index][0]
            errors.append(
                {
                    "image_path": image_path,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": float(
                        probabilities[batch_index, predicted_label].item()
                    ),
                    "cat_probability": float(
                        probabilities[batch_index, 0].item()
                    ),
                    "dog_probability": float(
                        probabilities[batch_index, 1].item()
                    ),
                }
            )

        sample_offset += labels.size(0)
        if batch_number % 20 == 0 or batch_number == len(loader):
            print(
                f"\r分析进度：{batch_number}/{len(loader)}",
                end="",
                flush=True,
            )

    print()
    errors.sort(key=lambda item: item["confidence"], reverse=True)
    return confusion_matrix, errors


def save_confusion_matrix(
    confusion_matrix: np.ndarray,
    output_path: Path,
) -> None:
    row_totals = confusion_matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        confusion_matrix,
        row_totals,
        out=np.zeros_like(confusion_matrix, dtype=np.float64),
        where=row_totals != 0,
    ) * 100.0

    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("真实类别内比例（%）")

    axis.set_title("猫狗分类混淆矩阵")
    axis.set_xlabel("预测类别")
    axis.set_ylabel("真实类别")
    axis.set_xticks((0, 1), ("猫", "狗"))
    axis.set_yticks((0, 1), ("猫", "狗"))

    threshold = 50.0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{confusion_matrix[row, column]:,}\n"
                f"{percentages[row, column]:.1f}%",
                ha="center",
                va="center",
                color="white" if percentages[row, column] > threshold else "black",
                fontsize=14,
                fontweight="medium",
            )

    total = int(confusion_matrix.sum())
    correct = int(np.trace(confusion_matrix))
    accuracy = correct / total if total else 0.0
    figure.text(
        0.5,
        0.01,
        f"测试样本：{total:,}    总体准确率：{accuracy * 100:.2f}%",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _select_balanced_errors(
    errors: list[dict],
    examples_per_direction: int,
) -> list[tuple[str, list[dict]]]:
    directions = []
    for true_label, predicted_label in ((0, 1), (1, 0)):
        selected = [
            item
            for item in errors
            if item["true_label"] == true_label
            and item["predicted_label"] == predicted_label
        ][:examples_per_direction]
        directions.append(
            (
                f"真实{CLASS_NAMES[true_label]} → 预测{CLASS_NAMES[predicted_label]}",
                selected,
            )
        )
    return directions


def save_error_examples(
    errors: list[dict],
    output_path: Path,
    examples_per_direction: int,
) -> None:
    columns = min(4, max(1, examples_per_direction))
    rows_per_direction = max(1, math.ceil(examples_per_direction / columns))
    total_rows = rows_per_direction * 2
    figure, axes = plt.subplots(
        total_rows,
        columns,
        figsize=(4 * columns, 3.8 * total_rows),
        squeeze=False,
    )

    directions = _select_balanced_errors(errors, examples_per_direction)
    for direction_index, (direction_name, selected) in enumerate(directions):
        row_start = direction_index * rows_per_direction
        figure.text(
            0.01,
            1 - (row_start + 0.45) / total_rows,
            direction_name,
            rotation=90,
            va="center",
            fontsize=13,
            fontweight="medium",
        )

        for position in range(rows_per_direction * columns):
            row = row_start + position // columns
            column = position % columns
            axis = axes[row, column]
            axis.axis("off")

            if position >= len(selected):
                if position == 0 and not selected:
                    axis.text(0.5, 0.5, "没有此类误判", ha="center", va="center")
                continue

            item = selected[position]
            with Image.open(item["image_path"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(
                f"预测置信度 {item['confidence'] * 100:.1f}%\n"
                f"{Path(item['image_path']).name}",
                fontsize=10,
            )

    figure.suptitle(
        "高置信度误判样本（越靠前表示模型越确信自己的错误判断）",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.035, 0, 1, 0.97))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_error_details(errors: list[dict], output_path: Path) -> None:
    fieldnames = (
        "image_path",
        "true_class",
        "predicted_class",
        "confidence",
        "cat_probability",
        "dog_probability",
    )
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in errors:
            image_path = Path(item["image_path"])
            try:
                saved_image_path = image_path.relative_to(PROJECT_ROOT)
            except ValueError:
                saved_image_path = image_path

            writer.writerow(
                {
                    "image_path": saved_image_path,
                    "true_class": CLASS_NAMES[item["true_label"]],
                    "predicted_class": CLASS_NAMES[item["predicted_label"]],
                    "confidence": f"{item['confidence']:.6f}",
                    "cat_probability": f"{item['cat_probability']:.6f}",
                    "dog_probability": f"{item['dog_probability']:.6f}",
                }
            )


def save_summary(
    confusion_matrix: np.ndarray,
    errors: list[dict],
    output_path: Path,
) -> None:
    total = int(confusion_matrix.sum())
    cat_total = int(confusion_matrix[0].sum())
    dog_total = int(confusion_matrix[1].sum())
    summary = {
        "total_samples": total,
        "correct_samples": int(np.trace(confusion_matrix)),
        "error_samples": len(errors),
        "accuracy": float(np.trace(confusion_matrix) / total),
        "cat_as_dog": int(confusion_matrix[0, 1]),
        "dog_as_cat": int(confusion_matrix[1, 0]),
        "cat_error_rate": float(confusion_matrix[0, 1] / cat_total),
        "dog_error_rate": float(confusion_matrix[1, 0] / dog_total),
    }
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.examples_per_direction <= 0:
        raise ValueError("examples-per-direction 必须大于 0")

    configure_chinese_font()
    device = select_device(args.device)
    dataset = ImageFolder(args.data_dir, transform=evaluation_transform())
    if dataset.class_to_idx != EXPECTED_CLASSES:
        raise ValueError(f"测试集类别映射错误：{dataset.class_to_idx}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    checkpoint = load_checkpoint(args.model_path)
    if checkpoint.get("class_to_idx") != dataset.class_to_idx:
        raise ValueError("模型类别映射与测试集不一致")

    model = CatDogCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    confusion_matrix, errors = collect_predictions(
        model, dataset, loader, device
    )

    matrix_path = args.output_dir / "confusion_matrix.png"
    examples_path = args.output_dir / "misclassified_examples.png"
    details_path = args.output_dir / "misclassified_samples.csv"
    summary_path = args.output_dir / "evaluation_summary.json"

    save_confusion_matrix(confusion_matrix, matrix_path)
    save_error_examples(
        errors,
        examples_path,
        args.examples_per_direction,
    )
    save_error_details(errors, details_path)
    save_summary(confusion_matrix, errors, summary_path)

    print(f"设备：{device}")
    print(f"误判样本：{len(errors)}/{len(dataset)}")
    print(f"混淆矩阵：{matrix_path}")
    print(f"误判样本图：{examples_path}")
    print(f"误判明细：{details_path}")
    print(f"评价摘要：{summary_path}")


if __name__ == "__main__":
    main()
