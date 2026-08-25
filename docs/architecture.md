# 猫狗分类器：设计与实验记录

中山大学 石全涛

## 任务

输入一张 $128\times128$ 的 RGB 图片，输出猫或狗。标签约定为：猫是 `0`，狗是 `1`。

## 数据

数据来自 [Kaggle Dogs vs. Cats](https://www.kaggle.com/datasets/princelv84/dogsvscats)。

| 目录 | 猫 | 狗 | 合计 |
| --- | ---: | ---: | ---: |
| 训练数据 | 10,000 | 10,000 | 20,000 |
| 测试数据 | 2,500 | 2,500 | 5,000 |

训练目录再按 8∶2 分成训练集和验证集，所以实际使用的是 16,000 张训练图片、4,000 张验证图片和 5,000 张测试图片。猫狗数量相同，不需要额外处理类别不平衡。

## 图片处理

原图大小不一。我先把图片转成 RGB，再从中心裁剪并缩放到 $128\times128$，训练时把像素值缩放到模型需要的范围。

选择 128 像素主要是为了控制手写卷积的计算量。这个尺寸能看清大部分轮廓和纹理，但中心裁剪也可能丢掉靠近边缘的主体。

PyTorch 训练时另外加入了随机裁剪和水平翻转。这样每轮看到的图片会有一点变化，可以减轻过拟合。

## 网络结构

三套代码使用的基础结构一致：

```text
输入
 ↓
Conv → ReLU → MaxPool
 ↓
Conv → ReLU → MaxPool
 ↓
Flatten → Linear
 ↓
猫 / 狗
```

| 层 | 参数 | 输出形状 |
| --- | --- | --- |
| 输入 | RGB 图片 | $3\times128\times128$ |
| 卷积 1 | 8 个 $3\times3$ 卷积核 | $8\times128\times128$ |
| 池化 1 | $2\times2$ | $8\times64\times64$ |
| 卷积 2 | 16 个 $3\times3$ 卷积核 | $16\times64\times64$ |
| 池化 2 | $2\times2$ | $16\times32\times32$ |
| 展平 | — | 16,384 |
| 全连接 | 两个类别分数 | 2 |

卷积的步长为 1、填充为 1，所以卷积前后的宽高不变。每次池化再把宽高减半。隐藏层使用 ReLU，最后用 Softmax 和交叉熵计算分类损失。

## 三套代码

### NumPy 手写版

卷积、池化、全连接、Softmax、反向传播和 SGD 都用 NumPy 完成。训练参数为：

- batch size：8；
- epoch：5；
- learning rate：0.01；
- optimizer：SGD。

模型保存到 `outputs/numpy_baseline/best_model_numpy.npz`。

### NumPy 模块化版

这个版本没有换网络，只把配置、训练循环、指标和存档分开，便于继续改动。模型单独保存在 `outputs/numpy_modular/best_model_numpy.npz`。

### PyTorch 版

网络仍是两层卷积，但训练改用 Adam，并加入归一化和数据增强：

- batch size：32；
- epoch：10；
- learning rate：0.001；
- optimizer：Adam；
- weight decay：0.0001。

程序支持 CUDA、Apple MPS 和 CPU。模型保存到 `outputs/pytorch/best_model_pytorch.pt`。

## 实验结果

NumPy 手写版训练 5 轮后的结果：

| 指标 | 数值 |
| --- | ---: |
| 最终训练损失 | 0.4132 |
| 最终训练准确率 | 80.99% |
| 最终验证损失 | 0.5001 |
| 最佳验证准确率 | 76.80% |
| 测试损失 | 0.5015 |
| 测试准确率 | 76.24% |
| 猫识别正确率 | 74.12% |
| 狗识别正确率 | 78.36% |
| 训练时间 | 678.73 秒 |

混淆矩阵中，真实猫有 1,853 张分对、647 张分成狗；真实狗有 1,959 张分对、541 张分成猫。数值记录在 [evaluation_summary.json](../outputs/numpy_baseline/evaluation_summary.json)。

PyTorch 版训练 10 轮，最好的验证结果出现在第 9 轮：

| 指标 | 数值 |
| --- | ---: |
| 最佳验证准确率 | 80.58% |
| 测试损失 | 0.4467 |
| 测试准确率 | 80.48% |
| 猫识别正确率 | 79.80% |
| 狗识别正确率 | 81.16% |

5,000 张测试图片中有 976 张判断错误，其中猫判成狗 505 张，狗判成猫 471 张。

我查看了 [误判样本](../outputs/pytorch/misclassified_examples.png)，错误主要出现在光线较暗、主体较小、遮挡、局部特写和多动物画面中。测试集里还有 Logo、无动物图片和 “No Photo Available” 之类的异常样本，它们也会影响结果。完整明细在 [misclassified_samples.csv](../outputs/pytorch/misclassified_samples.csv)。

PyTorch 版比手写版高 4.24 个百分点，不过两边的优化器、归一化和数据增强都不同，所以不能把差距简单归因于框架。

## 接下来想做的改动

- 清理异常图片；
- 记录训练损失和准确率曲线；
- 比较 SGD、Adam 和 AdamW；
- 尝试 BatchNorm、Dropout 和全局平均池化；
- 网络加深后再测试残差连接；
- 尝试预训练 ResNet。
