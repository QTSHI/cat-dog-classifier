"""分类指标。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    confusion_matrix: np.ndarray | None = None

    def class_accuracy(self, label):
        """计算一个类别的准确率。"""

        if self.confusion_matrix is None:
            raise ValueError("本次评估没有计算混淆矩阵")

        class_total = int(np.sum(self.confusion_matrix[label]))
        if class_total == 0:
            return 0.0

        return float(self.confusion_matrix[label, label] / class_total)


def add_to_confusion_matrix(confusion_matrix, labels, predictions):
    """把当前 batch 计入混淆矩阵。"""

    np.add.at(confusion_matrix, (labels, predictions), 1)
