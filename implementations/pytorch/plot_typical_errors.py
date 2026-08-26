"""从误判明细中整理有代表性的错误样本。"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TYPICAL_SAMPLES = (
    "data/processed/test/cats/cat.4688.jpg",
    "data/processed/test/dogs/dog.8898.jpg",
    "data/processed/test/dogs/dog.7602.jpg",
    "data/processed/test/dogs/dog.5804.jpg",
    "data/processed/test/cats/cat.11432.jpg",
    "data/processed/test/dogs/dog.9913.jpg",
)


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
    parser = argparse.ArgumentParser(description="绘制典型误判样本")
    parser.add_argument(
        "--details-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pytorch" / "misclassified_samples.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "pytorch"
            / "typical_misclassified_samples.png"
        ),
    )
    return parser.parse_args()


def load_selected_errors(details_path: Path) -> list[dict[str, str]]:
    with details_path.open(encoding="utf-8-sig", newline="") as file:
        rows = {row["image_path"]: row for row in csv.DictReader(file)}

    missing = [path for path in TYPICAL_SAMPLES if path not in rows]
    if missing:
        raise ValueError(f"误判明细中找不到样本：{', '.join(missing)}")
    return [rows[path] for path in TYPICAL_SAMPLES]


def save_figure(samples: list[dict[str, str]], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11, 7.2), squeeze=False)

    for number, (axis, sample) in enumerate(zip(axes.flat, samples), start=1):
        image_path = PROJECT_ROOT / sample["image_path"]
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))

        confidence = float(sample["confidence"]) * 100
        axis.set_title(
            f"{number}. 真实{sample['true_class']} → 预测{sample['predicted_class']}"
            f"（{confidence:.1f}%）\n{image_path.name}",
            fontsize=11,
        )
        axis.axis("off")

    figure.suptitle("典型误判样本", fontsize=16, fontweight="medium")
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    samples = load_selected_errors(args.details_path)
    save_figure(samples, args.output_path)
    print(f"典型误判样本图：{args.output_path}")


if __name__ == "__main__":
    main()
