"""设备、随机种子和模型文件。"""

import random
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前环境无法使用 CUDA")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("当前环境无法使用 Apple MPS")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    validation_accuracy: float,
    class_to_idx: dict[str, int],
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
            "class_to_idx": class_to_idx,
            "config": serializable_config,
        },
        path,
    )


def load_checkpoint(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 PyTorch 模型文件：{path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
