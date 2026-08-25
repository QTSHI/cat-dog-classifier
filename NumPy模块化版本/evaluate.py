"""评估 NumPy 模型并输出混淆矩阵。"""

import argparse
from pathlib import Path

import numpy as np

from checkpoint import load_parameters
from config import OUTPUT_ROOT, PREPROCESSED_DATA_ROOT
from data_loader import collect_dataset
from engine import evaluate_model, limit_dataset
from model import CatVsDog


def parse_args():
    parser = argparse.ArgumentParser(description="评估 NumPy 猫狗分类模型")
    parser.add_argument(
        "--data-dir", type=Path, default=PREPROCESSED_DATA_ROOT / "test"
    )
    parser.add_argument(
        "--model-path", type=Path, default=OUTPUT_ROOT / "best_model.npz"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def evaluate(data_dir, model_path, batch_size=16, max_samples=None):
    """评估指定模型并返回 ClassificationMetrics。"""

    test_paths, test_labels = collect_dataset(data_dir)
    test_paths, test_labels = limit_dataset(test_paths, test_labels, max_samples)

    print("测试集总数：", len(test_paths))
    print("猫的数量：", np.sum(test_labels == 0))
    print("狗的数量：", np.sum(test_labels == 1))

    model = load_parameters(CatVsDog(seed=42), model_path)
    print("已加载模型：", model_path)

    metrics = evaluate_model(
        model,
        test_paths,
        test_labels,
        batch_size=batch_size,
        include_confusion_matrix=True,
        progress_label="测试进度",
        progress_interval=50,
    )
    matrix = metrics.confusion_matrix

    print(f"测试损失：{metrics.loss:.4f}")
    print(f"测试准确率：{metrics.accuracy * 100:.2f}%")
    print()
    print("混淆矩阵：")
    print("          预测猫  预测狗")
    print(f"真实猫：{matrix[0, 0]:6d} {matrix[0, 1]:6d}")
    print(f"真实狗：{matrix[1, 0]:6d} {matrix[1, 1]:6d}")
    print()
    print(f"猫识别正确率：{metrics.class_accuracy(0) * 100:.2f}%")
    print(f"狗识别正确率：{metrics.class_accuracy(1) * 100:.2f}%")

    if int(np.sum(matrix)) != len(test_labels):
        raise RuntimeError("混淆矩阵样本数与测试集不一致")

    return metrics


def main():
    args = parse_args()
    evaluate(args.data_dir, args.model_path, args.batch_size, args.max_samples)


if __name__ == "__main__":
    main()
