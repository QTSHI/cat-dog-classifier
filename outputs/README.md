# 输出文件

这里保留历史实验材料，供 README、设计文档与 Notebook 展示。`numpy_baseline` 保存手写版测试记录；`pytorch` 保存 PyTorch 模型、训练曲线、混淆矩阵和误判图片。数据集不在仓库中。

| 文件 | 来源与用途 |
| --- | --- |
| `numpy_baseline/evaluation_summary.json` | 从历史终端输出整理的参数与指标；当前训练、测试脚本不会自动写出该摘要 |
| `numpy_baseline/best_model_numpy.npz` | NumPy 训练脚本生成的最佳参数；被忽略，未提交，需自行训练 |
| `pytorch/best_model_pytorch.pt` | 已提交的 PyTorch 最佳模型，用于单图预测、测试与特征可视化 |
| `pytorch/training_history.json` | 每轮 loss 和准确率；来自 2026 年 8 月 26 日按相同参数重跑的记录 |
| `pytorch/training_loss_curve.png` | 对应 `training_history.json` 的训练、验证 loss 曲线 |
| `pytorch/evaluation_summary.json` | 测试样本数、正确与错误数量、准确率等汇总 |
| `pytorch/misclassified_samples.csv` | 每个误判样本的相对路径、类别与预测概率 |
| `pytorch/confusion_matrix.png`、`misclassified_examples.png` | 混淆矩阵与误判图片 |
| `pytorch/typical_misclassified_samples.png` | 设计文档使用的 6 个典型误判样本 |
| `pytorch/first_layer_filters.png` | 最佳模型的第一层卷积核 |

历史训练曲线没有覆盖原来的最佳模型，因此曲线与 README 中最佳验证准确率不属于同一次逐轮记录。保留这一点才能正确比较实验，详情见 [设计与实验记录](../docs/architecture.md#训练损失曲线)。

## 重新生成

在仓库根目录、安装好 PyTorch 版依赖后运行。误判分析需要 `data/processed/test` 中的图片；典型误判图还需要脚本指定的原始样本文件名保持一致。

```bash
python implementations/pytorch/visualize_errors.py --device auto
python implementations/pytorch/plot_typical_errors.py
python implementations/pytorch/plot_filters.py
python implementations/pytorch/plot_history.py
```

`visualize_errors.py` 生成测试摘要、误判 CSV、混淆矩阵与误判图；其余脚本分别生成典型误判图、卷积核图与训练曲线。`evaluate.py` 只把指标打印到终端。

这些命令默认覆盖 `outputs/pytorch` 中的同名文件。新实验建议训练时使用 `--output-dir outputs/local-run`，并为后续分析指定对应模型、输入文件和输出路径（见各脚本 `--help`）。`outputs/local-run/` 已被忽略，公开的历史材料不需要每次试跑后更新。
