"""预测一张图片。"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from data import evaluation_transform
from model import CatDogCNN
from utils import load_checkpoint, select_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预测单张猫狗图片")
    parser.add_argument("image", type=Path, help="待预测图片路径")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pytorch" / "best_model_pytorch.pt",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"找不到图片：{args.image}")

    device = select_device(args.device)
    checkpoint = load_checkpoint(args.model_path)
    model = CatDogCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with Image.open(args.image) as image:
        image = image.convert("RGB")
        image_tensor = evaluation_transform()(image).unsqueeze(0).to(device)

    probabilities = torch.softmax(model(image_tensor), dim=1)[0].cpu()
    predicted_label = int(probabilities.argmax().item())
    label_to_chinese = {0: "猫", 1: "狗"}

    print(f"预测结果：{label_to_chinese[predicted_label]}")
    print(f"猫的概率：{probabilities[0].item() * 100:.2f}%")
    print(f"狗的概率：{probabilities[1].item() * 100:.2f}%")


if __name__ == "__main__":
    main()
