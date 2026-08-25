"""NumPy 模型共用的训练和评估流程。"""

import numpy as np

from data import batch_iterator
from metrics import ClassificationMetrics, add_to_confusion_matrix


def limit_dataset(image_paths, labels, max_samples):
    """调试时只保留前 max_samples 个样本。"""

    if max_samples is None:
        return image_paths, labels

    sample_count = min(max_samples, len(image_paths))
    return image_paths[:sample_count], labels[:sample_count]


def train_one_epoch(
    model,
    optimizer,
    image_paths,
    labels,
    batch_size,
    random_generator,
    epoch=None,
    total_epochs=None,
    progress_interval=20,
):
    """训练一轮并返回平均损失和准确率。"""

    if len(image_paths) == 0:
        raise ValueError("训练数据不能为空")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    iterator = batch_iterator(
        image_paths,
        labels,
        batch_size=batch_size,
        random_generator=random_generator,
        shuffle=True,
    )

    for batch_number, (batch_images, batch_labels) in enumerate(iterator, start=1):
        gradients = model.gradient(batch_images, batch_labels)
        batch_loss = float(model.loss_layer.loss)
        predictions = np.argmax(model.loss_layer.probabilities, axis=1)
        optimizer.update(model.params, gradients)

        current_batch_size = len(batch_labels)
        total_loss += batch_loss * current_batch_size
        total_correct += int(np.sum(predictions == batch_labels))
        total_samples += current_batch_size

        should_report = (
            progress_interval
            and (batch_number % progress_interval == 0 or batch_number == total_batches)
        )
        if should_report:
            epoch_text = ""
            if epoch is not None and total_epochs is not None:
                epoch_text = f"Epoch {epoch}/{total_epochs} "
            print(
                f"\r{epoch_text}Batch {batch_number}/{total_batches}",
                end="",
                flush=True,
            )

    return ClassificationMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


def evaluate_model(
    model,
    image_paths,
    labels,
    batch_size,
    include_confusion_matrix=False,
    progress_label=None,
    progress_interval=0,
):
    """只做前向传播，可选计算混淆矩阵。"""

    if len(image_paths) == 0:
        raise ValueError("评估数据不能为空")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    confusion_matrix = None

    if include_confusion_matrix:
        confusion_matrix = np.zeros((2, 2), dtype=np.int64)

    iterator = batch_iterator(
        image_paths,
        labels,
        batch_size=batch_size,
        random_generator=np.random.default_rng(seed=0),
        shuffle=False,
    )

    for batch_number, (batch_images, batch_labels) in enumerate(iterator, start=1):
        batch_loss = model.loss(batch_images, batch_labels)
        predictions = np.argmax(model.loss_layer.probabilities, axis=1)

        current_batch_size = len(batch_labels)
        total_loss += float(batch_loss) * current_batch_size
        total_correct += int(np.sum(predictions == batch_labels))
        total_samples += current_batch_size

        if confusion_matrix is not None:
            add_to_confusion_matrix(confusion_matrix, batch_labels, predictions)

        should_report = (
            progress_label
            and progress_interval
            and (batch_number % progress_interval == 0 or batch_number == total_batches)
        )
        if should_report:
            print(
                f"\r{progress_label}：{batch_number}/{total_batches}",
                end="",
                flush=True,
            )

    if progress_label and progress_interval:
        print()

    return ClassificationMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        confusion_matrix=confusion_matrix,
    )
