# PyTorch 版

这个版本把手写网络换成 PyTorch 写法，仍然保留两层卷积，便于和 NumPy 结果比较。训练使用 Adam、随机裁剪和水平翻转。

## 文件

- `data.py`：数据增强、数据划分和 DataLoader；
- `model.py`：网络；
- `engine.py`：训练和测试循环；
- `train.py`：训练脚本；
- `evaluate.py`：测试脚本；
- `predict.py`：预测单张图片；
- `visualize_errors.py`：混淆矩阵和误判图片；
- `plot_typical_errors.py`：整理文档中使用的典型误判样本；
- `plot_filters.py`：显示训练后的第一层卷积核；
- `plot_history.py`：训练损失曲线；
- `utils.py`：设备、随机种子和模型存取。

## 训练

使用支持 PyTorch 的 Python 环境（代码包含 Python 3.10+ 类型注解语法），在仓库根目录运行。先按 [数据准备说明](../../data/README.md) 生成 `data/processed/train/{cats,dogs}` 和 `data/processed/test/{cats,dogs}`。

```bash
python -m pip install -r implementations/pytorch/requirements.txt
python implementations/pytorch/train.py --epochs 10 --batch-size 32 --output-dir outputs/local-run
```

默认从训练目录按类别划分 20% 验证集，使用随机种子 `23`、Adam、学习率 `0.001`、weight decay `0.0001`。`--device auto` 会依次选择 CUDA、Apple MPS、CPU；也可以指定 `--device cpu`。训练、测试脚本的 `--num-workers` 默认是 `0`。

支持 MPS 的 Mac 可以调整 batch size 和数据加载进程数，例如：

```bash
python -u implementations/pytorch/train.py \
  --device mps \
  --batch-size 64 \
  --num-workers 4 \
  --output-dir outputs/local-run
```

上面的多行命令使用 Bash 续行符；PowerShell 中请写成一行。可通过 `python implementations/pytorch/train.py --help` 查看数据目录、输出目录和少量样本试跑选项。

## 测试

评估新训练的模型：

```bash
python implementations/pytorch/evaluate.py --model-path outputs/local-run/best_model_pytorch.pt --device auto
```

使用仓库保留的模型预测单张图片（将路径换成自己的照片）：

```bash
python implementations/pytorch/predict.py "path/to/example.jpg" --device cpu
```

单图预测不需要完整数据集。`evaluate.py` 需要预处理后的测试集，只在终端打印指标；`visualize_errors.py` 才会生成误判明细与测试摘要。

## 可视化与输出

下面的命令使用仓库保留的模型和实验文件；误判分析需要本地测试图片，典型误判图还依赖代码中指定的六张图片：

```bash
python implementations/pytorch/visualize_errors.py --device auto
python implementations/pytorch/plot_typical_errors.py
python implementations/pytorch/plot_filters.py
python implementations/pytorch/plot_history.py
```

默认模型和图表目录为 `outputs/pytorch`。训练会写入 `best_model_pytorch.pt`、`training_history.json` 和 `training_loss_curve.png`；上面的可视化命令也会覆盖同名图表或摘要。保留历史结果时，请分别设置训练的 `--output-dir`、误判分析的 `--model-path` / `--output-dir`、绘图脚本的输入与 `--output-path` 参数，具体见各脚本 `--help`。

仓库保留的历史测试准确率是 **80.48%**，指标来源和重新生成方式见 [输出文件说明](../../outputs/README.md)。训练曲线来自单独一次重跑，不与原始最佳模型的逐轮指标混用。
