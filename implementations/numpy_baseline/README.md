# NumPy 手写版

这是最早写的版本。卷积、池化、反向传播和 SGD 都没有调用深度学习框架。

## 文件

- `preprocess.py`：裁剪图片；
- `data.py`：读取图片、划分数据和组成 batch；
- `layers.py`：各个网络层；
- `model.py`：把网络层连起来；
- `optimizer.py`：SGD；
- `smoke_test.py`：用四张图片测试反向传播；
- `train.py`：训练；
- `evaluate.py`：测试和混淆矩阵。

## 运行

```bash
python3 -m pip install -r implementations/numpy_baseline/requirements.txt
python3 implementations/numpy_baseline/smoke_test.py
python3 implementations/numpy_baseline/train.py
python3 implementations/numpy_baseline/evaluate.py
```

训练结果保存在 `outputs/numpy_baseline`。目前测试准确率是 76.24%。
