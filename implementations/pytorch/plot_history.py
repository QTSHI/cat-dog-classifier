"""根据训练记录绘制 loss 曲线。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_font():
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_training_loss(history, output_path):
    """画出每一轮的训练 loss 和验证 loss。"""

    if not history:
        raise ValueError("训练记录为空")

    configure_font()
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    validation_losses = [item["validation_loss"] for item in history]

    figure, axis = plt.subplots(figsize=(8, 5.2))
    axis.plot(
        epochs,
        train_losses,
        marker="o",
        linewidth=2,
        label="训练集",
    )
    axis.plot(
        epochs,
        validation_losses,
        marker="o",
        linewidth=2,
        label="验证集",
    )
    axis.set_title("PyTorch 训练损失曲线")
    axis.set_xlabel("训练轮数（epoch）")
    axis.set_ylabel("交叉熵损失（loss）")
    axis.set_xticks(epochs)
    axis.grid(alpha=0.25)
    axis.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="绘制训练 loss 曲线")
    parser.add_argument(
        "--history-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pytorch" / "training_history.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pytorch" / "training_loss_curve.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    history = json.loads(args.history_path.read_text(encoding="utf-8"))
    plot_training_loss(history, args.output_path)
    print("训练曲线：", args.output_path)


if __name__ == "__main__":
    main()
