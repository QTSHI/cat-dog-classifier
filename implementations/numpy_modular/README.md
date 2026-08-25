# NumPy 模块化版

这个版本保留 NumPy 手写 CNN，同时把配置、训练、指标和模型存档拆成可复用模块。

## 文件

- `config.py`：路径和默认参数；
- `preprocess.py`：图片预处理；
- `data.py`：数据读取和划分；
- `layers.py`、`model.py`：网络结构；
- `optimizer.py`：SGD；
- `engine.py`：训练和评估流程；
- `metrics.py`：指标和混淆矩阵；
- `checkpoint.py`：模型存档；
- `train.py`、`evaluate.py`：运行入口；
- `smoke_test.py`：快速检查训练链路。

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

最佳模型保存在 `outputs/numpy_modular/best_model_numpy.npz`，不会覆盖原始 NumPy 版模型。
