import time
from pathlib import Path

import numpy as np

from data_loader import (
    batch_iterator,
    collect_dataset,
    stratified_split,
)
from model import CatVsDog
from optimizer import SGD


# -------------------------
# 训练配置
# -------------------------

SEED = 23
VALIDATION_RATIO = 0.2 #验证数据的比例 
BATCH_SIZE = 8
LEARNING_RATE = 0.01
EPOCHS = 5

# 第一次调试只使用少量数据
# 确认程序正常后改成 None
MAX_TRAIN_SAMPLES = None
MAX_VALIDATION_SAMPLES = None


def limit_dataset(
    image_paths,
    labels,
    max_samples,
):
    """调试时限制数据数量。"""

    if max_samples is None:
        return image_paths, labels

    max_samples = min(
        max_samples,
        len(image_paths),
    )

    return (
        image_paths[:max_samples],
        labels[:max_samples],
    )


def evaluate(
    model,
    image_paths,
    labels,
    batch_size,
):
    """
    在验证集上计算平均损失和准确率。

    验证过程只做前向传播，不更新参数。
    """

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 验证时不需要随机打乱
    validation_rng = np.random.default_rng(
        seed=0
    )

    iterator = batch_iterator(
        image_paths,
        labels,
        batch_size=batch_size,
        random_generator=validation_rng,
        shuffle=False,
    )

    for batch_images, batch_labels in iterator:
        loss = model.loss(
            batch_images,
            batch_labels,
        )

        predictions = np.argmax(
            model.loss_layer.probabilities,
            axis=1,
        )

        current_batch_size = len(batch_labels)

        total_loss += (
            float(loss)
            * current_batch_size
        )

        total_correct += np.sum(
            predictions == batch_labels
        )

        total_samples += current_batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def save_parameters(model, save_path):
    """把所有模型参数保存为一个 npz 文件。"""

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        save_path,
        **model.params,
    )


def main():
    project_root = (
        Path(__file__).resolve().parents[1]
    )

    dataset_root = (
        project_root
        / "数据"
        / "Preprocessed_Data"
        / "train"
    )

    output_dir = project_root / "输出"
    best_model_path = (
        output_dir
        / "best_model.npz"
    )

    rng = np.random.default_rng(
        seed=SEED
    )

    # 收集全部数据
    image_paths, labels = collect_dataset(
        dataset_root
    )

    # 分层划分训练集和验证集
    (
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
    ) = stratified_split(
        image_paths,
        labels,
        validation_ratio=VALIDATION_RATIO,
        random_generator=rng,
    )

    # 调试阶段限制样本数
    train_paths, train_labels = limit_dataset(
        train_paths,
        train_labels,
        MAX_TRAIN_SAMPLES,
    )

    (
        validation_paths,
        validation_labels,
    ) = limit_dataset(
        validation_paths,
        validation_labels,
        MAX_VALIDATION_SAMPLES,
    )

    print(
        "训练集：",
        len(train_paths),
        "验证集：",
        len(validation_paths),
    )

    print(
        "训练集猫/狗：",
        np.sum(train_labels == 0),
        "/",
        np.sum(train_labels == 1),
    )

    print(
        "验证集猫/狗：",
        np.sum(validation_labels == 0),
        "/",
        np.sum(validation_labels == 1),
    )

    model = CatVsDog(seed=SEED)

    optimizer = SGD(
        learning_rate=LEARNING_RATE
    )

    best_validation_accuracy = -1.0
    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_iterator = batch_iterator(
            train_paths,
            train_labels,
            batch_size=BATCH_SIZE,
            random_generator=rng,
            shuffle=True,
        )

        total_batches = (
            len(train_paths)
            + BATCH_SIZE
            - 1
        ) // BATCH_SIZE

        for batch_number, (
            batch_images,
            batch_labels,
        ) in enumerate(
            train_iterator,
            start=1,
        ):
            # gradient 内部会先完成一次前向传播
            gradients = model.gradient(
                batch_images,
                batch_labels,
            )

            # 读取本次前向传播的损失和概率
            batch_loss = float(
                model.loss_layer.loss
            )

            predictions = np.argmax(
                model.loss_layer.probabilities,
                axis=1,
            )

            # 使用梯度更新全部参数
            optimizer.update(
                model.params,
                gradients,
            )

            current_batch_size = len(
                batch_labels
            )

            total_loss += (
                batch_loss
                * current_batch_size
            )

            total_correct += np.sum(
                predictions == batch_labels
            )

            total_samples += current_batch_size

            if (
                batch_number % 20 == 0
                or batch_number == total_batches
            ):
                print(
                    f"\rEpoch {epoch}/{EPOCHS} "
                    f"Batch {batch_number}/"
                    f"{total_batches}",
                    end="",
                    flush=True,
                )

        train_loss = (
            total_loss / total_samples
        )

        train_accuracy = (
            total_correct / total_samples
        )

        validation_loss, validation_accuracy = (
            evaluate(
                model,
                validation_paths,
                validation_labels,
                BATCH_SIZE,
            )
        )

        epoch_seconds = (
            time.time() - epoch_start
        )

        print()
        print(
            f"训练损失：{train_loss:.4f}，"
            f"训练准确率："
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"验证损失："
            f"{validation_loss:.4f}，"
            f"验证准确率："
            f"{validation_accuracy * 100:.2f}%"
        )

        print(
            f"本轮耗时："
            f"{epoch_seconds:.2f} 秒"
        )

        # 只保存验证准确率最高的模型
        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            save_parameters(
                model,
                best_model_path,
            )

            print(
                "已保存最佳模型：",
                best_model_path,
            )

    total_seconds = (
        time.time() - training_start
    )

    print()
    print(
        "训练完成，最佳验证准确率："
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        f"总耗时：{total_seconds:.2f} 秒"
    )


if __name__ == "__main__":
    main()