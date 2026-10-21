# Notebook

`cat_dog_classifier_workflow.ipynb` 使用 PyTorch 完成猫狗分类实验，可以按顺序查看数据、模型、训练曲线、特征图、混淆矩阵、误判样本和卷积核。

在项目根目录运行：

```bash
python -m pip install -r implementations/pytorch/requirements.txt -r notebooks/requirements.txt
jupyter notebook notebooks/cat_dog_classifier_workflow.ipynb
```

`notebooks/requirements.txt` 只列出 Notebook 相关依赖，需要同时安装 PyTorch 版依赖。用 VS Code 打开也可以，选择同一个装有 PyTorch、torchvision 和 Matplotlib 的 Python 环境；PyTorch 脚本使用 Python 3.10+ 语法。

按顺序运行还需要按 [数据准备说明](../data/README.md) 放好训练、测试图片。Notebook 默认读取仓库已有的 `outputs/pytorch/best_model_pytorch.pt`，不会自动重新训练；部分单元格仍会读取本地图片来显示样本和特征图。`RUN_TEST_EVALUATION = False` 时读取已保存的测试摘要，改为 `True` 后才重新执行测试集评估。

Notebook 使用的历史训练曲线与原始最佳模型来自不同训练记录，说明见 [输出文件](../outputs/README.md)。
