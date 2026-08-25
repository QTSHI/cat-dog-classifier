# Cat and Dog Classifier

这是一个猫狗图片分类项目，保留了 NumPy 原始实现、NumPy 模块化实现和 PyTorch 实现，方便对照学习。

项目设计和实验结果见 [architecture.md](docs/architecture.md)。

## 项目结构

```text
cat-dog-classifier/
├── implementations/
│   ├── numpy_baseline/    # 原始 NumPy 手写版
│   ├── numpy_modular/     # 模块化 NumPy 版
│   └── pytorch/           # PyTorch 版
├── notebooks/             # Jupyter Notebook
├── docs/                  # 设计文档
├── data/
│   ├── raw/               # 原始图片
│   └── processed/         # 128×128 预处理图片
└── outputs/
    ├── numpy_baseline/
    ├── numpy_modular/
    └── pytorch/
```

数据集图片不会上传到 GitHub，需要放在 `data/raw` 或 `data/processed` 对应目录中。

## NumPy 原始版

```bash
python3 -m pip install -r implementations/numpy_baseline/requirements.txt
python3 implementations/numpy_baseline/smoke_test.py
python3 implementations/numpy_baseline/train.py
python3 implementations/numpy_baseline/evaluate.py
```

详细说明见 [numpy_baseline/README.md](implementations/numpy_baseline/README.md)。

## NumPy 模块化版

```bash
python3 -m pip install -r implementations/numpy_modular/requirements.txt
python3 implementations/numpy_modular/smoke_test.py
python3 implementations/numpy_modular/train.py
python3 implementations/numpy_modular/evaluate.py
```

详细说明见 [numpy_modular/README.md](implementations/numpy_modular/README.md)。

## PyTorch 版

```bash
python3 -m pip install -r implementations/pytorch/requirements.txt
python3 implementations/pytorch/train.py --device mps
python3 implementations/pytorch/evaluate.py --device mps
python3 implementations/pytorch/visualize_errors.py --device mps --num-workers 4
```

更多参数见 [pytorch/README.md](implementations/pytorch/README.md)。

## 当前结果

| 版本 | 最佳验证准确率 | 测试准确率 |
| --- | ---: | ---: |
| NumPy 原始版 | 76.80% | 76.24% |
| PyTorch 版 | 80.58% | 80.48% |
