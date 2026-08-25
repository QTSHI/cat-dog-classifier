"""读写模型参数。"""

from pathlib import Path

import numpy as np


def save_parameters(model, save_path):
    """保存参数。"""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **model.params)


def load_parameters(model, model_path):
    """加载参数。"""

    model.load_parameters(model_path)
    return model
