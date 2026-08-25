from pathlib import Path

import numpy as np
from PIL import Image

VALID_SUFFIXES = {".jpg",".jpeg",".png"}

def collect_dataset(dataset_root):
    class_to_label = {
        "cats":0,
        "dogs":1,
    }
    image_path = []
    labels = []

    for class_name,label in class_to_label.items():
        class_dir = dataset_root / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"找不到类别目录：{class_dir}"
            )

        class_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in VALID_SUFFIXES
        )

        image_path.extend(class_paths)
        labels.extend([label] * len(class_paths))

    return (
        np.array(image_path,dtype=object),
        np.array(labels,dtype=np.int64)
    )

def load_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        image_array = (
            np.asarray(
                image,
                dtype=np.float32,
            )
            /255.0
        )

    image_array = image_array.transpose(2,0,1)

    if image_array.shape != (3,128,128):
        raise ValueError(
            f"{image_path} 的形状错误："
            f"{image_array.shape}"
        )       
    return image_array

def load_batch(batch_paths,batch_labels):
    images = [
        load_image(path)
        for path in batch_paths
    ]

    batch_images = np.stack(
        images,
        axis=0,
    )

    batch_labels=np.asarray(
        batch_labels,
        dtype=np.int64,
    )

    return batch_images,batch_labels

def batch_iterator(
    image_paths,
    labels,
    batch_size,
    random_generator,
    shuffle=True,
):
    """
    逐批读取数据，避免把全部图片放进内存。
    """

    indices = np.arange(len(image_paths))

    if shuffle:
        random_generator.shuffle(indices)

    for start in range(
        0,
        len(indices),
        batch_size,
    ):
        end = start + batch_size
        batch_indices = indices[start:end]

        yield load_batch(
            image_paths[batch_indices],
            labels[batch_indices],
        )

def stratified_split(
    image_paths,
    labels,
    validation_ratio,
    random_generator,
):
    """
    分别划分猫和狗，确保训练集、验证集类别平衡。
    """

    train_indices = []
    validation_indices = []

    for label in np.unique(labels):
        class_indices = np.flatnonzero(
            labels == label
        )

        random_generator.shuffle(class_indices)

        validation_count = int(
            len(class_indices)
            * validation_ratio
        )

        validation_indices.extend(
            class_indices[:validation_count]
        )

        train_indices.extend(
            class_indices[validation_count:]
        )

    train_indices = np.array(
        train_indices,
        dtype=np.int64,
    )

    validation_indices = np.array(
        validation_indices,
        dtype=np.int64,
    )

    # 避免训练数据仍然是先猫后狗
    random_generator.shuffle(train_indices)
    random_generator.shuffle(validation_indices)

    return (
        image_paths[train_indices],
        labels[train_indices],
        image_paths[validation_indices],
        labels[validation_indices],
    )