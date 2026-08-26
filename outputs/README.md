# 输出文件

`numpy_baseline` 保存手写版模型和测试记录，`pytorch` 保存 PyTorch 模型、训练曲线、混淆矩阵和误判图片。

`pytorch/training_history.json` 记录每轮 loss 和准确率，`pytorch/training_loss_curve.png` 是对应的训练、验证 loss 曲线。

`pytorch/typical_misclassified_samples.png` 是设计文档中使用的 6 个典型误判样本。

`pytorch/first_layer_filters.png` 显示最佳模型训练后的第一层卷积核。

PyTorch 的误判分析可以重新运行：

```bash
python3 implementations/pytorch/visualize_errors.py --device mps --num-workers 4
```
