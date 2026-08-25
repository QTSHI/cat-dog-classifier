# NumPy 模块化版本说明

这个版本不依赖 PyTorch，主要目的是自己实现 CNN 的计算过程。

## 文件作用

建议按照下面的顺序阅读：

1. `config.py`：项目路径和默认训练参数。
2. `batch_preprocess.py`：把原始图片裁剪成 $128\times128$。
3. `data_loader.py`：读取图片、划分数据、生成批次。
4. `layers.py`：卷积、池化、ReLU、展平和全连接层。
5. `model.py`：把各层连接成完整 CNN。
6. `optimizer.py`：使用 SGD 更新参数。
7. `engine.py`：通用的单轮训练和模型评估。
8. `metrics.py`：保存损失、准确率和混淆矩阵。
9. `checkpoint.py`：保存和加载模型参数。
10. `test.py`：用少量真实图片检查损失能否下降。
11. `train.py`：训练入口，只负责组织流程。
12. `evaluate.py`：测试入口，输出准确率和混淆矩阵。

这样改模型时主要看 `model.py`，改训练流程时看 `engine.py`，改参数时看 `config.py`，不需要把所有内容堆在 `train.py` 里。

## 运行顺序

如果预处理数据已经存在，不需要再次运行 `batch_preprocess.py`。

先检查训练链路：

```bash
python "NumPy模块化版本/test.py"
```

再进行训练和测试：

```bash
python "NumPy模块化版本/train.py"
python "NumPy模块化版本/evaluate.py"
```

想先用少量图片试跑，可以直接传参数，不用修改源码：

```bash
python "NumPy模块化版本/train.py" --epochs 1 --max-train-samples 80 --max-validation-samples 20
python "NumPy模块化版本/evaluate.py" --max-samples 100
```

查看全部可选参数：

```bash
python "NumPy模块化版本/train.py" --help
```

新训练得到的模型保存在 `NumPy模块化版本/输出/best_model.npz`，不会覆盖原来 `输出` 文件夹中的模型。重新预处理图片时，也会默认写入本文件夹里的 `预处理数据`。

## 自己优化时的顺序

建议一次只改一项，每次记录验证准确率和训练时间：

1. 清理无动物、Logo、严重遮挡等异常图片；
2. 增加随机翻转或随机裁剪；
3. 尝试 Adam 或带动量的 SGD；
4. 增加一个卷积层，确认普通深层网络能正常训练；
5. 最后再尝试残差连接。

残差连接需要保证相加的两个张量形状相同。当前模型很浅，所以它不是必须项，也不一定比清理数据和改进优化器更有效。

## 当前结果

本次使用完整数据训练 5 轮，终端记录为：

- 最终训练损失：0.4132；
- 最终训练准确率：80.99%；
- 最佳验证准确率：76.80%；
- 测试损失：0.5015；
- 测试准确率：76.24%；
- 猫识别正确率：74.12%；
- 狗识别正确率：78.36%；
- 总训练时间：678.73 秒。

详细数据保存在 `输出/numpy_evaluation_summary.json`。
