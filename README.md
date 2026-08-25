# 猫狗分类器

这是我学习卷积神经网络时做的猫狗分类项目。最开始用 NumPy 手写卷积和反向传播，之后整理了一份模块化代码，又写了一个 PyTorch 版本作对照。

## 目录

```text
implementations/
├── numpy_baseline/    最早的 NumPy 手写版
├── numpy_modular/     拆分过的 NumPy 版
└── pytorch/           PyTorch 版
notebooks/             Notebook 学习记录
docs/                  结构和实验记录
data/                  本地数据集
outputs/               模型与测试结果
```

模型结构、参数选择和实验结果写在 [architecture.md](docs/architecture.md) 中。

## 数据

数据来自 [Kaggle Dogs vs. Cats](https://www.kaggle.com/datasets/princelv84/dogsvscats)。图片没有提交到仓库，目录格式见 [data/README.md](data/README.md)。

## 运行

NumPy 手写版：

```bash
python3 -m pip install -r implementations/numpy_baseline/requirements.txt
python3 implementations/numpy_baseline/smoke_test.py
python3 implementations/numpy_baseline/train.py
python3 implementations/numpy_baseline/evaluate.py
```

NumPy 模块化版：

```bash
python3 -m pip install -r implementations/numpy_modular/requirements.txt
python3 implementations/numpy_modular/smoke_test.py
python3 implementations/numpy_modular/train.py
python3 implementations/numpy_modular/evaluate.py
```

PyTorch 版：

```bash
python3 -m pip install -r implementations/pytorch/requirements.txt
python3 implementations/pytorch/train.py --device mps
python3 implementations/pytorch/evaluate.py --device mps
```

## 结果

| 代码 | 最佳验证准确率 | 测试准确率 |
| --- | ---: | ---: |
| NumPy 手写版 | 76.80% | 76.24% |
| PyTorch 版 | 80.58% | 80.48% |

模块化 NumPy 版只整理了代码，没有改变网络和计算方法，因此没有把它当成新的对比实验。
