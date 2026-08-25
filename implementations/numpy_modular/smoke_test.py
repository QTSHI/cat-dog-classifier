"""用四张图片检查前向和反向传播。"""

import numpy as np

from config import PREPROCESSED_DATA_ROOT
from data import collect_dataset, load_batch
from model import CatVsDog
from optimizer import SGD


def build_sample_batch(dataset_root, samples_per_class=2, seed=42):
    image_paths, labels = collect_dataset(dataset_root)
    print("图片总数：", len(image_paths))
    print("猫的数量：", np.sum(labels == 0))
    print("狗的数量：", np.sum(labels == 1))

    selected_indices = np.concatenate(
        [
            np.flatnonzero(labels == 0)[:samples_per_class],
            np.flatnonzero(labels == 1)[:samples_per_class],
        ]
    )
    expected_count = samples_per_class * 2
    if len(selected_indices) != expected_count:
        raise ValueError("猫或狗的样本数量不足")

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(selected_indices)
    return load_batch(image_paths[selected_indices], labels[selected_indices])


def overfit_small_batch(steps=10, learning_rate=0.005):
    """反复训练同一个 batch，观察损失是否下降。"""

    batch_images, batch_labels = build_sample_batch(
        PREPROCESSED_DATA_ROOT / "train"
    )
    print("Batch 图片形状：", batch_images.shape)
    print("Batch 标签：", batch_labels)

    model = CatVsDog(seed=42)
    optimizer = SGD(learning_rate=learning_rate)
    initial_loss = float(model.loss(batch_images, batch_labels))
    print("初始损失：", initial_loss)

    for step in range(1, steps + 1):
        gradients = model.gradient(batch_images, batch_labels)
        optimizer.update(model.params, gradients)
        current_loss = float(model.loss(batch_images, batch_labels))
        print(f"第 {step:02d} 次更新，损失：{current_loss:.6f}")

    final_loss = float(model.loss(batch_images, batch_labels))
    print("最终损失：", final_loss)

    assert batch_images.shape == (4, 3, 128, 128)
    assert batch_labels.shape == (4,)
    assert final_loss < initial_loss
    return initial_loss, final_loss


def main():
    overfit_small_batch()
    print("真实图片 SGD 更新测试通过！")


if __name__ == "__main__":
    main()
