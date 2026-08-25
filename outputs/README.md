# 输出文件

`numpy_baseline` 保存手写版模型和测试记录，`numpy_modular` 保存模块化版本的模型，`pytorch` 保存 PyTorch 模型、混淆矩阵和误判图片。

PyTorch 的误判分析可以重新运行：

```bash
python3 implementations/pytorch/visualize_errors.py --device mps --num-workers 4
```
