"""批量裁剪并统一猫狗图片尺寸。"""

import argparse
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from config import MODULE_PREPROCESSED_DATA_ROOT, RAW_DATA_ROOT


VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def preprocess_image(source_path, target_path, image_size=(128, 128)):
    """处理一张图片。"""

    with Image.open(source_path) as image:
        image = image.convert("RGB")
        processed_image = ImageOps.fit(
            image,
            image_size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        processed_image.save(target_path, format="JPEG", quality=95)


def preprocess_dataset(
    source_root,
    target_root,
    image_size=(128, 128),
    max_images=None,
    class_names=("cats", "dogs"),
):
    """处理整个数据集，并返回每个类别的成功/失败数量。"""

    source_root = Path(source_root)
    target_root = Path(target_root)
    summary = {}

    for class_name in class_names:
        source_class_dir = source_root / class_name
        target_class_dir = target_root / class_name
        if not source_class_dir.is_dir():
            raise FileNotFoundError(f"找不到类别目录：{source_class_dir}")

        target_class_dir.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(
            path
            for path in source_class_dir.iterdir()
            if path.suffix.lower() in VALID_SUFFIXES
        )
        if max_images is not None:
            image_paths = image_paths[:max_images]

        success_count = 0
        failed_count = 0
        for index, source_path in enumerate(image_paths, start=1):
            target_path = target_class_dir / f"{source_path.stem}.jpg"
            try:
                preprocess_image(source_path, target_path, image_size)
                success_count += 1
            except (UnidentifiedImageError, OSError) as error:
                failed_count += 1
                print(f"处理失败：{source_path}，原因：{error}")

            if index % 500 == 0:
                print(f"{class_name} 已处理 {index}/{len(image_paths)}")

        summary[class_name] = {"success": success_count, "failed": failed_count}
        print(f"{class_name} 完成：成功 {success_count} 张，失败 {failed_count} 张")

    print("预处理数据保存位置：", target_root)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="批量预处理猫狗图片")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = args.source_dir or RAW_DATA_ROOT / args.split
    target_root = args.target_dir or MODULE_PREPROCESSED_DATA_ROOT / args.split
    preprocess_dataset(
        source_root,
        target_root,
        image_size=(args.image_size, args.image_size),
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
