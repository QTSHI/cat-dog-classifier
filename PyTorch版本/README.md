# PyTorch 猫狗分类器

这是原 NumPy 手写版本的独立 PyTorch 改写版，不会覆盖或依赖 `代码` 目录中的实现。

## 目录说明

- `model.py`：CNN 模型。
- `data.py`：数据增强、分层划分和 DataLoader。
- `engine.py`：训练与评估循环。
- `train.py`：训练入口，使用 Adam 并保存验证准确率最高的模型。
- `evaluate.py`：测试集准确率和混淆矩阵。
- `visualize_errors.py`：混淆矩阵图片和高置信度误判样本分析。
- `predict.py`：单张图片预测。

默认数据目录沿用原项目：

```text
数据/Preprocessed_Data/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

## 安装依赖

在项目根目录执行：

```bash
python -m pip install -r "PyTorch版本/requirements.txt"
```

## 快速检查

先用少量数据确认完整训练链路：

```bash
python "PyTorch版本/train.py" \
  --epochs 1 \
  --max-train-samples 64 \
  --max-validation-samples 32
```

## 完整训练

```bash
python "PyTorch版本/train.py" --epochs 10 --batch-size 32
```

程序按 `CUDA → Apple MPS → CPU` 的顺序自动选择设备。最佳模型默认保存至：

```text
输出/PyTorch/best_model.pt
```

如需强制使用某个设备，可增加 `--device cpu`、`--device mps` 或 `--device cuda`。

## 测试

```bash
python "PyTorch版本/evaluate.py" --device mps
```

## 预测单张图片

```bash
python "PyTorch版本/predict.py" "/完整路径/图片.jpg"
```

## 查看误判情况

```bash
python "PyTorch版本/visualize_errors.py"
```

运行后会在 `输出/PyTorch` 中生成：

- `confusion_matrix.png`：每类预测数量和比例；
- `misclassified_examples.png`：模型最有信心但仍然判断错误的图片；
- `misclassified_samples.csv`：全部误判图片及猫、狗概率；
- `evaluation_summary.json`：总体准确率和分类别错误率。

## 常用训练参数

```bash
python "PyTorch版本/train.py" \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --validation-ratio 0.2 \
  --num-workers 0
```

macOS 初次运行建议保留 `--num-workers 0`。如内存不足，可把 `--batch-size` 改为 16 或 8。

如果小规模测试可以正常运行，完整训练时可以使用：

```bash
python3 -u "PyTorch版本/train.py" \
  --device mps \
  --batch-size 64 \
  --num-workers 4
```

## 当前结果

当前 `best_model.pt` 在第 9 轮取得最高验证准确率，测试结果为：

- 最佳验证准确率：80.58%；
- 测试损失：0.4467；
- 测试准确率：80.48%；
- 猫识别正确率：79.80%；
- 狗识别正确率：81.16%。

测试集中共有 976 张误判图片，可以通过 `misclassified_examples.png` 和 `misclassified_samples.csv` 继续检查。
