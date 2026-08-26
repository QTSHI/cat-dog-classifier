"""显示训练后第一层卷积核。"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    parser = argparse.ArgumentParser(description="显示第一层卷积核")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pytorch" / "best_model_pytorch.pt",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pytorch" / "first_layer_filters.png",
    )
    return parser.parse_args()


def load_first_layer_filters(model_path: Path) -> np.ndarray:
    checkpoint = load_checkpoint(model_path)
    weights = checkpoint["model_state_dict"]["features.0.weight"]
    if weights.ndim != 4 or weights.shape[1] != 3:
        raise ValueError(f"第一层卷积核形状不符合预期：{tuple(weights.shape)}")
    return weights.detach().cpu().numpy()


def filters_to_rgb(filters: np.ndarray) -> np.ndarray:
    """把有正负值的 RGB 权重映射到 0～1，0 显示为中灰色。"""

    scale = float(np.abs(filters).max())
    if scale == 0:
        return np.full(filters.transpose(0, 2, 3, 1).shape, 0.5)
    rgb_filters = 0.5 + filters.transpose(0, 2, 3, 1) / (2 * scale)
    return np.clip(rgb_filters, 0.0, 1.0)


def save_figure(filters: np.ndarray, output_path: Path) -> None:
    rgb_filters = filters_to_rgb(filters)
    norms = np.linalg.norm(filters.reshape(filters.shape[0], -1), axis=1)
    figure, axes = plt.subplots(2, 4, figsize=(10, 5.8), squeeze=False)

    for number, (axis, rgb_filter, norm) in enumerate(
        zip(axes.flat, rgb_filters, norms),
        start=1,
    ):
        axis.imshow(rgb_filter, interpolation="nearest")
        axis.set_title(f"Filter {number}  ‖W‖={norm:.3f}", fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle("第一层卷积核（8 × 3 × 3 × 3）", fontsize=16)
    figure.text(
        0.5,
        0.015,
        "统一尺度显示：中灰色接近 0，颜色变化表示 RGB 通道的正负权重",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=2.5, w_pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    filters = load_first_layer_filters(args.model_path)
    save_figure(filters, args.output_path)
    print(f"第一层卷积核：{args.output_path}")


if __name__ == "__main__":
    main()
