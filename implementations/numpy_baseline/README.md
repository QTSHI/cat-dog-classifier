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

在仓库根目录执行。先按 [数据准备说明](../../data/README.md) 生成 `data/processed/train/{cats,dogs}` 和 `data/processed/test/{cats,dogs}`，图片必须为 `128 × 128` RGB。

```bash
python -m pip install -r implementations/numpy_baseline/requirements.txt
python implementations/numpy_baseline/smoke_test.py
python implementations/numpy_baseline/train.py
python implementations/numpy_baseline/evaluate.py
```

`smoke_test.py` 读取前两张猫图和前两张狗图，在同一个 batch 上更新 10 次，检查形状与损失是否下降；它需要真实图片，不能在空数据目录直接运行。

训练参数位于 `train.py` 文件顶部，默认随机种子 `23`、验证比例 `0.2`、batch size `8`、学习率 `0.01`、训练 `5` 轮。这一版没有命令行参数；调试时可修改 `MAX_TRAIN_SAMPLES`、`MAX_VALIDATION_SAMPLES` 限制样本数量。

训练结果保存在 `outputs/numpy_baseline/best_model_numpy.npz`，按最高验证准确率保存。该文件被 `.gitignore` 忽略，没有包含在仓库中，所以必须先训练再运行 `evaluate.py`。测试脚本读取 `data/processed/test` 并在终端打印指标和混淆矩阵。

仓库保留的历史测试准确率为 **76.24%**，来源见 [evaluation_summary.json](../../outputs/numpy_baseline/evaluation_summary.json)。这个摘要由历史终端输出记录整理，当前 `evaluate.py` 不会自动生成该 JSON。
