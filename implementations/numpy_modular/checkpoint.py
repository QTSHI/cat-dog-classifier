"""模型参数存取。"""

from pathlib import Path

import numpy as np


def save_parameters(model, save_path):
    """把模型参数保存成一个 npz 文件。"""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **model.params)


def load_parameters(model, model_path):
    """加载参数并返回模型，方便连续调用。"""

    model.load_parameters(model_path)
    return model
