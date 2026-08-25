"""NumPy 版本共用的路径和默认配置。"""

from dataclasses import dataclass
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parents[1]
PREPROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "numpy_modular"


@dataclass(frozen=True)
class TrainingConfig:
    """训练入口使用的配置；命令行参数可以覆盖这些默认值。"""

    seed: int = 23
    validation_ratio: float = 0.2
    batch_size: int = 8
    learning_rate: float = 0.01
    epochs: int = 5
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    data_dir: Path = PREPROCESSED_DATA_ROOT / "train"
    output_dir: Path = OUTPUT_ROOT

    @property
    def best_model_path(self):
        return self.output_dir / "best_model_numpy.npz"
