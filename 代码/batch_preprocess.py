from pathlib import Path
from PIL import Image, ImageOps,UnidentifiedImageError

IMAGE_SIZE = (128,128)

MAX_IMAGES = None

# 当前要处理的数据集
DATASET_SPLIT = "test"

# 当前项目根目录：猫狗分类器
project_root = Path(__file__).resolve().parents[1]

# 原始测试集：
# 猫狗分类器/数据/原始数据/test
source_root = (
    project_root
    / "数据"
    / "原始数据"
    / DATASET_SPLIT
)

# 预处理测试集：
# 猫狗分类器/数据/Preprocessed_Data/test
target_root = (
    project_root
    / "数据"
    / "Preprocessed_Data"
    / DATASET_SPLIT
)

class_names = ["cats","dogs"]
valid_suffixes = {".jpg",".jpeg",".png"}

def preprocess_image(source_path,target_path):

    with Image.open(source_path) as image:
        image = image.convert("RGB")

    processed_image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
        centering=(0.5,0.5),
    )

    processed_image.save(
        target_path,
        format = "JPEG",
        quality = 95
    )

for class_name in class_names:
    source_class_dir = source_root/class_name
    target_class_dir = target_root/class_name

    target_class_dir.mkdir(parents=True,exist_ok=True)

    image_paths = sorted(
        path
        for path in source_class_dir.iterdir()
        if path.suffix.lower() in valid_suffixes
    )

    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    success_count = 0
    failed_count = 0

    for index,source_path in enumerate(image_paths,start=1):
        target_path = target_class_dir/f"{source_path.stem}.jpg"

        try:
            preprocess_image(source_path,target_path)
            success_count +=1
        except(UnidentifiedImageError,OSError) as error:
            failed_count +=1
            print(f"处理失败：{source_path}，原因:{error}")

        if index % 500 ==0:
            print(f"{class_name} 已处理 {index}/{len(image_paths)}")
    print(
        f"{class_name} 完成："
        f"成功 {success_count} 张，失败 {failed_count} 张"
    )

print("预处理数据保存位置：", target_root)
