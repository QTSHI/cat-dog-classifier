# NumPy 模块化版

这份代码和手写版使用同一个网络，区别只是把重复内容拆开了。之后想换优化器、增加指标或调整训练过程时，不用都挤在 `train.py` 里修改。

## 文件

- `config.py`：路径和训练参数；
- `preprocess.py`、`data.py`：图片和数据；
- `layers.py`、`model.py`：网络；
- `optimizer.py`：SGD；
- `engine.py`：一轮训练和测试；
- `metrics.py`：准确率与混淆矩阵；
- `checkpoint.py`：参数存取；
- `train.py`、`evaluate.py`：运行脚本；
- `smoke_test.py`：四张图片的小测试。

## 运行

```bash
python3 -m pip install -r implementations/numpy_modular/requirements.txt
python3 implementations/numpy_modular/smoke_test.py
python3 implementations/numpy_modular/train.py
python3 implementations/numpy_modular/evaluate.py
```

少量数据试跑：

```bash
python3 implementations/numpy_modular/train.py \
  --epochs 1 \
  --max-train-samples 80 \
  --max-validation-samples 20
```

模型写入 `outputs/numpy_modular/best_model_numpy.npz`，不会覆盖手写版模型。
