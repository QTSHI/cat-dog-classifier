from pathlib import Path

import numpy as np

from data import collect_dataset, load_batch
from model import CatVsDog
from optimizer import SGD


project_root = Path(__file__).resolve().parents[2]

dataset_root = (
    project_root
    / "data"
    / "processed"
    / "train"
)

image_paths, labels = collect_dataset(
    dataset_root
)

print("图片总数：", len(image_paths))
print("猫的数量：", np.sum(labels == 0))
print("狗的数量：", np.sum(labels == 1))


# 选择两张猫图和两张狗图
cat_indices = np.flatnonzero(labels == 0)[:2]
dog_indices = np.flatnonzero(labels == 1)[:2]

batch_indices = np.concatenate(
    [cat_indices, dog_indices]
)

# 打乱这四张图片的顺序
rng = np.random.default_rng(seed=42)
rng.shuffle(batch_indices)

batch_images, batch_labels = load_batch(
    image_paths[batch_indices],
    labels[batch_indices],
)

print("Batch 图片形状：", batch_images.shape)
print("Batch 标签：", batch_labels)


model = CatVsDog(seed=42)

optimizer = SGD(
    learning_rate=0.005
)

initial_loss = model.loss(
    batch_images,
    batch_labels,
)

print("初始损失：", initial_loss)


# 反复学习同一个 mini-batch
for step in range(1, 11):
    gradients = model.gradient(
        batch_images,
        batch_labels,
    )

    optimizer.update(
        model.params,
        gradients,
    )

    current_loss = model.loss(
        batch_images,
        batch_labels,
    )

    print(
        f"第 {step:02d} 次更新，"
        f"损失：{current_loss:.6f}"
    )


final_loss = model.loss(
    batch_images,
    batch_labels,
)

print("最终损失：", final_loss)

assert batch_images.shape == (4, 3, 128, 128)
assert batch_labels.shape == (4,)
assert final_loss < initial_loss

print("真实图片 SGD 更新测试通过！")
