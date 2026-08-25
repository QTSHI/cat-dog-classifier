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
- `utils.py`：设备、随机种子和模型存取。

## 训练

```bash
python3 -m pip install -r implementations/pytorch/requirements.txt
python3 implementations/pytorch/train.py --epochs 10 --batch-size 32
```

Mac 可以使用 MPS：

```bash
python3 -u implementations/pytorch/train.py \
  --device mps \
  --batch-size 64 \
  --num-workers 4
```

## 测试

```bash
python3 implementations/pytorch/evaluate.py --device mps
python3 implementations/pytorch/predict.py "/图片路径/example.jpg" --device mps
python3 implementations/pytorch/visualize_errors.py --device mps --num-workers 4
```

模型和图表写入 `outputs/pytorch`。目前测试准确率是 80.48%。
