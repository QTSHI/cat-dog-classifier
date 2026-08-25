# 输出文件

## `numpy_baseline`

- `best_model_numpy.npz`：NumPy 原始版最佳模型；
- `evaluation_summary.json`：训练和测试结果。

## `numpy_modular`

- `best_model_numpy.npz`：NumPy 模块化版最佳模型，训练后生成。

## `pytorch`

- `best_model_pytorch.pt`：PyTorch 最佳模型；
- `confusion_matrix.png`：测试集混淆矩阵；
- `misclassified_examples.png`：高置信度误判图片；
- `misclassified_samples.csv`：全部误判样本和概率；
- `evaluation_summary.json`：测试结果。

重新生成 PyTorch 评估结果：

```bash
python3 implementations/pytorch/evaluate.py --device mps
python3 implementations/pytorch/visualize_errors.py --device mps --num-workers 4
```
