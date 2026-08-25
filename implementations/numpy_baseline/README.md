# NumPy 原始版

这个版本手动实现卷积、池化、反向传播和 SGD，主要用于理解 CNN 的底层计算过程。

## 文件

1. `preprocess.py`：统一图片尺寸。
2. `data.py`：读取图片、划分数据和生成批次。
3. `layers.py`：卷积、池化、ReLU、展平和全连接层。
4. `model.py`：完整 CNN。
5. `optimizer.py`：SGD 参数更新。
6. `smoke_test.py`：用四张图片检查损失能否下降。
7. `train.py`：完整训练。
8. `evaluate.py`：测试准确率和混淆矩阵。

## 运行

```bash
python3 -m pip install -r implementations/numpy_baseline/requirements.txt
python3 implementations/numpy_baseline/smoke_test.py
python3 implementations/numpy_baseline/train.py
python3 implementations/numpy_baseline/evaluate.py
```

最佳模型保存在 `outputs/numpy_baseline/best_model_numpy.npz`。

## 当前结果

- 最佳验证准确率：76.80%；
- 测试损失：0.5015；
- 测试准确率：76.24%；
- 猫识别正确率：74.12%；
- 狗识别正确率：78.36%；
- 总训练时间：678.73 秒。

详细结果见 [evaluation_summary.json](../../outputs/numpy_baseline/evaluation_summary.json)。
