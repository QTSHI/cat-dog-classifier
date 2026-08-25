"""训练 NumPy 手写 CNN，并保存验证集表现最好的参数。"""

import argparse
import time
from pathlib import Path

import numpy as np

from checkpoint import save_parameters
from config import TrainingConfig
from data_loader import collect_dataset, stratified_split
from engine import evaluate_model, limit_dataset, train_one_epoch
from model import CatVsDog
from optimizer import SGD


def parse_args():
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="训练 NumPy 猫狗分类模型")
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--validation-ratio", type=float, default=defaults.validation_ratio)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    return parser.parse_args()


def build_config(args):
    return TrainingConfig(
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


def train(config):
    """按配置训练模型，返回最佳验证准确率。"""

    rng = np.random.default_rng(seed=config.seed)
    image_paths, labels = collect_dataset(config.data_dir)

    train_paths, train_labels, validation_paths, validation_labels = stratified_split(
        image_paths,
        labels,
        validation_ratio=config.validation_ratio,
        random_generator=rng,
    )
    train_paths, train_labels = limit_dataset(
        train_paths, train_labels, config.max_train_samples
    )
    validation_paths, validation_labels = limit_dataset(
        validation_paths, validation_labels, config.max_validation_samples
    )

    print("训练集：", len(train_paths), "验证集：", len(validation_paths))
    print("训练集猫/狗：", np.sum(train_labels == 0), "/", np.sum(train_labels == 1))
    print(
        "验证集猫/狗：",
        np.sum(validation_labels == 0),
        "/",
        np.sum(validation_labels == 1),
    )

    model = CatVsDog(seed=config.seed)
    optimizer = SGD(learning_rate=config.learning_rate)
    best_validation_accuracy = -1.0
    training_start = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        training_metrics = train_one_epoch(
            model,
            optimizer,
            train_paths,
            train_labels,
            batch_size=config.batch_size,
            random_generator=rng,
            epoch=epoch,
            total_epochs=config.epochs,
        )
        validation_metrics = evaluate_model(
            model,
            validation_paths,
            validation_labels,
            batch_size=config.batch_size,
        )

        print()
        print(
            f"训练损失：{training_metrics.loss:.4f}，"
            f"训练准确率：{training_metrics.accuracy * 100:.2f}%"
        )
        print(
            f"验证损失：{validation_metrics.loss:.4f}，"
            f"验证准确率：{validation_metrics.accuracy * 100:.2f}%"
        )
        print(f"本轮耗时：{time.time() - epoch_start:.2f} 秒")

        if validation_metrics.accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_metrics.accuracy
            save_parameters(model, config.best_model_path)
            print("已保存最佳模型：", config.best_model_path)

    print()
    print(f"训练完成，最佳验证准确率：{best_validation_accuracy * 100:.2f}%")
    print(f"总耗时：{time.time() - training_start:.2f} 秒")
    return best_validation_accuracy


def main():
    train(build_config(parse_args()))


if __name__ == "__main__":
    main()
