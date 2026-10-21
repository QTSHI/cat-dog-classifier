# 数据放置方式

数据来自 [Kaggle Dogs vs. Cats](https://www.kaggle.com/datasets/princelv84/dogsvscats)。请自行下载并遵守数据来源的使用条件；图片没有上传到仓库。两个实现都默认读取 `processed`，本地按下面的目录放置：

```text
data/
├── raw/
│   ├── train/cats
│   ├── train/dogs
│   ├── test/cats
│   └── test/dogs
└── processed/
    ├── train/cats
    ├── train/dogs
    ├── test/cats
    └── test/dogs
```

`raw` 放原图，`processed` 放裁剪为 $128\times128$ 的图片。每个 split 下必须有 `cats` 和 `dogs` 两个目录，不能直接把所有图片放在 split 根目录。

## 预处理

预处理脚本是 [implementations/numpy_baseline/preprocess.py](../implementations/numpy_baseline/preprocess.py)，处理后的图片供两个实现共用。它将图片转换为 RGB，中心裁剪并缩放为 `128 × 128`，输出 JPEG；支持读取 `.jpg`、`.jpeg`、`.png`。

该脚本使用文件顶部的配置，**没有命令行参数**。当前 `DATASET_SPLIT = "test"`，直接运行只会处理测试集。

在仓库根目录执行：

```bash
python -m pip install -r implementations/numpy_baseline/requirements.txt
python implementations/numpy_baseline/preprocess.py
```

完整准备步骤：

1. 将训练、测试原图分别放到对应的 `data/raw/<split>/cats` 和 `dogs` 目录。
2. 将脚本中的 `DATASET_SPLIT` 设为 `"train"`，运行上面的预处理命令。
3. 将 `DATASET_SPLIT` 设为 `"test"`，再运行一次。`MAX_IMAGES = None` 表示处理全部图片；填数字时会限制每类处理数量。
4. 确认四个 `data/processed/<split>/<class>` 目录均有图片。预处理会打印成功、失败数量，无法读取的图片会被跳过。

`train.py` 从训练目录按类别划分验证集（默认 20%），不需要单独建立 `validation` 目录。`smoke_test.py` 至少需要训练目录中两张猫图、两张狗图。

重新处理同名文件会覆盖已有图片；脚本不会清理旧文件。更换数据集或使用 `MAX_IMAGES` 时，请先核对目标目录内容，避免混入之前的样本。

两版训练均使用固定随机种子，但依赖版本、设备与数据内容会影响复现。历史实验的数据数量与指标见 [设计与实验记录](../docs/architecture.md)。
