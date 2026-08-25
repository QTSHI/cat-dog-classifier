# 输出文件说明

## NumPy 版

- `best_model.npz`：验证准确率最高的 NumPy 模型参数。
- `numpy_evaluation_summary.json`：NumPy 训练与测试结果。

## PyTorch 版

- `best_model.pt`：验证准确率最高的 PyTorch 模型；
- `confusion_matrix.png`：测试集混淆矩阵；
- `misclassified_examples.png`：高置信度误判图片；
- `misclassified_samples.csv`：全部误判样本和预测概率；
- `evaluation_summary.json`：测试准确率及分类别错误率。

训练新模型后，可以重新执行：

```bash
python3 "PyTorch版本/evaluate.py" --device mps
python3 "PyTorch版本/visualize_errors.py" --device mps --num-workers 4
```
