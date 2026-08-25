# Jupyter Notebook 版本

此文件夹保留了 `猫狗分类器/代码` 中全部 Python 文件对应的 Notebook 版本，原始 `.py` 文件未被修改。

如果希望在一个文件中完成全部流程，直接打开 `CNN猫狗分类器_完整流程.ipynb`。

也可以从 `00_项目导航.ipynb` 开始，依次学习数据预处理、数据加载、神经网络层、优化器、CNN 模型、训练链路测试、完整训练和独立测试集评估。

## 启动方法

推荐直接使用 VS Code 打开 `.ipynb`，并选择安装了 NumPy 和 Pillow 的 Conda Python 环境。

如需在浏览器中使用 Jupyter Notebook，先安装 Notebook 服务：

```bash
cd "/Users/shiquantao/Desktop/深度学习/猫狗分类器/Jupyter_Notebook"
python -m pip install notebook
jupyter notebook
```

如果使用 VS Code，可以直接打开任意 `.ipynb` 文件并选择当前 Conda Python 环境。

## 重新生成 Notebook

原始 Python 文件更新后，可运行：

```bash
python build_notebooks.py
```

这会重新生成所有 Notebook，并覆盖旧的 Notebook 版本。
