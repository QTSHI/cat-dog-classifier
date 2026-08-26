# Notebook

`cat_dog_classifier_workflow.ipynb` 使用 PyTorch 完成猫狗分类实验，可以按顺序查看数据、模型、训练曲线、特征图、混淆矩阵、误判样本和卷积核。

在项目根目录运行：

```bash
python3 -m pip install -r notebooks/requirements.txt
jupyter notebook notebooks/cat_dog_classifier_workflow.ipynb
```

用 VS Code 打开也可以，记得选择装有 PyTorch、torchvision 和 Matplotlib 的 Python 环境。Notebook 默认读取已有模型，不会自动重新训练。
