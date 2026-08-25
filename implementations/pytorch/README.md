# PyTorch 版

这是独立的 PyTorch 实现，支持自动求导、数据增强、Adam，以及 CUDA、Apple MPS 和 CPU。

## 文件

- `data.py`：数据增强、分层划分和 DataLoader；
- `model.py`：CNN 模型；
- `engine.py`：训练与评估循环；
- `train.py`：训练入口；
- `evaluate.py`：测试准确率和混淆矩阵；
- `predict.py`：单张图片预测；
- `visualize_errors.py`：混淆矩阵和误判样本分析；
- `utils.py`：设备、随机种子和模型存档。

## 安装

```bash
python3 -m pip install -r implementations/pytorch/requirements.txt
```

## 训练

```bash
python3 implementations/pytorch/train.py --epochs 10 --batch-size 32
```

Mac 可明确指定 MPS：

```bash
python3 -u implementations/pytorch/train.py \
  --device mps \
  --batch-size 64 \
  --num-workers 4
```

最佳模型保存在 `outputs/pytorch/best_model_pytorch.pt`。

## 测试与预测

```bash
python3 implementations/pytorch/evaluate.py --device mps
python3 implementations/pytorch/predict.py "/完整路径/图片.jpg" --device mps
python3 implementations/pytorch/visualize_errors.py --device mps --num-workers 4
```

可视化结果保存在 `outputs/pytorch`。

## 当前结果

- 最佳验证准确率：80.58%；
- 测试损失：0.4467；
- 测试准确率：80.48%；
- 猫识别正确率：79.80%；
- 狗识别正确率：81.16%。
