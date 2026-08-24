from pathlib import Path

import numpy as np

from data_loader import (
    batch_iterator,
    collect_dataset,
)
from model import CatVsDog


BATCH_SIZE = 16


def main():
    project_root = (
        Path(__file__).resolve().parents[1]
    )

    test_root = (
        project_root
        / "数据"
        / "Preprocessed_Data"
        / "test"
    )

    model_path = (
        project_root
        / "输出"
        / "best_model.npz"
    )

    # 收集测试图片和标签
    test_paths, test_labels = collect_dataset(
        test_root
    )

    print("测试集总数：", len(test_paths))
    print(
        "猫的数量：",
        np.sum(test_labels == 0),
    )
    print(
        "狗的数量：",
        np.sum(test_labels == 1),
    )

    # 创建相同结构的网络
    model = CatVsDog(seed=42)

    # 使用训练保存的最佳参数覆盖随机参数
    model.load_parameters(model_path)

    print("已加载模型：", model_path)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 行表示真实类别，列表示预测类别
    #
    #           预测猫  预测狗
    # 真实猫
    # 真实狗
    confusion_matrix = np.zeros(
        (2, 2),
        dtype=np.int64,
    )

    random_generator = np.random.default_rng(
        seed=0
    )

    test_iterator = batch_iterator(
        test_paths,
        test_labels,
        batch_size=BATCH_SIZE,
        random_generator=random_generator,
        shuffle=False,
    )

    total_batches = (
        len(test_paths)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    for batch_number, (
        batch_images,
        batch_labels,
    ) in enumerate(
        test_iterator,
        start=1,
    ):
        # 只做前向传播，不更新参数
        batch_loss = model.loss(
            batch_images,
            batch_labels,
        )

        predictions = np.argmax(
            model.loss_layer.probabilities,
            axis=1,
        )

        current_batch_size = len(
            batch_labels
        )

        total_loss += (
            float(batch_loss)
            * current_batch_size
        )

        total_correct += np.sum(
            predictions == batch_labels
        )

        total_samples += current_batch_size

        # 把每个样本加入混淆矩阵
        np.add.at(
            confusion_matrix,
            (batch_labels, predictions),
            1,
        )

        if (
            batch_number % 50 == 0
            or batch_number == total_batches
        ):
            print(
                f"\r测试进度："
                f"{batch_number}/{total_batches}",
                end="",
                flush=True,
            )

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    cat_accuracy = (
        confusion_matrix[0, 0]
        / np.sum(confusion_matrix[0])
    )

    dog_accuracy = (
        confusion_matrix[1, 1]
        / np.sum(confusion_matrix[1])
    )

    print()
    print(
        f"测试损失：{average_loss:.4f}"
    )
    print(
        f"测试准确率：{accuracy * 100:.2f}%"
    )

    print()
    print("混淆矩阵：")
    print("          预测猫  预测狗")
    print(
        f"真实猫："
        f"{confusion_matrix[0, 0]:6d} "
        f"{confusion_matrix[0, 1]:6d}"
    )
    print(
        f"真实狗："
        f"{confusion_matrix[1, 0]:6d} "
        f"{confusion_matrix[1, 1]:6d}"
    )

    print()
    print(
        f"猫识别正确率："
        f"{cat_accuracy * 100:.2f}%"
    )
    print(
        f"狗识别正确率："
        f"{dog_accuracy * 100:.2f}%"
    )

    assert total_samples == 5000
    assert np.sum(confusion_matrix) == 5000


if __name__ == "__main__":
    main()