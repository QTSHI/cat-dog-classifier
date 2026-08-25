# 猫狗分类器

这个项目有三个版本，原文件都保留着：

- `代码`：NumPy 手写版，用来学习卷积、反向传播和 SGD；
- `PyTorch版本`：框架版，用来快速训练、测试和分析错误；
- `Jupyter_Notebook`：把手写版整理成一个 Notebook，方便边看边运行。

项目思路和实验结果见 [结构设计.md](框架设计/结构设计.md)。

## 文件夹

```text
猫狗分类器/
├── 代码/                 NumPy 手写版
├── PyTorch版本/          PyTorch 版
├── Jupyter_Notebook/     Notebook 版
├── 数据/                 原始数据和预处理数据
├── 输出/                 模型、图表和测试结果
└── 框架设计/             项目说明
```

## NumPy 版本

```bash
python "代码/train.py"
python "代码/evaluate.py"
```

每个文件的用途见 [NumPy 版说明](代码/README.md)。

## PyTorch 版本

安装依赖：

```bash
python3 -m pip install -r "PyTorch版本/requirements.txt"
```

使用 Mac GPU 训练：

```bash
python3 -u "PyTorch版本/train.py" \
  --device mps \
  --batch-size 64 \
  --num-workers 4
```

测试和生成误判图：

```bash
python3 "PyTorch版本/evaluate.py" --device mps
python3 "PyTorch版本/visualize_errors.py" --device mps --num-workers 4
```

更多参数见 [PyTorch 版说明](PyTorch版本/README.md)。

## 当前结果

NumPy 手写版：

- 最佳验证准确率：76.80%；
- 测试准确率：76.24%；
- 猫识别正确率：74.12%；
- 狗识别正确率：78.36%。

PyTorch 版：

- 最佳验证准确率：80.58%；
- 测试准确率：80.48%；
- 猫识别正确率：79.80%；
- 狗识别正确率：81.16%。

