# 开发与协作

这是学习 CNN 的个人实验项目。修改时请保留 NumPy 手写实现与 PyTorch 实现各自的学习价值，并说明改动影响哪一版。

## 提交问题

请提供操作系统、Python 版本、相关依赖版本、设备（CPU / CUDA / MPS）、执行命令和完整报错。数据问题请说明目录结构与图片尺寸；需要样本时使用可公开分享的图片，避免上传完整数据集或本机私有路径。

## 检查改动

在仓库根目录执行无需数据的语法检查：

```bash
python -m compileall -q implementations
git diff --check
```

GitHub Actions 的轻量检查还会解析已跟踪的 JSON 和 Notebook JSON。它不安装训练依赖，不加载模型，也不验证准确率；通过检查只代表相应的文件语法可解析。

修改 NumPy 训练流程后，准备每类至少两张训练图片，再运行：

```bash
python implementations/numpy_baseline/smoke_test.py
```

修改 PyTorch 训练流程后，可在准备好的数据上做小规模检查，并将输出与历史结果分开：

```bash
python implementations/pytorch/train.py --device cpu --epochs 1 --max-train-samples 32 --max-validation-samples 16 --output-dir outputs/local-run
```

小规模检查不能用来更新 README 中的完整测试指标。若修改模型、数据处理或训练配置，请记录数据来源与划分、随机种子、参数、实际依赖版本和验证结果；没有执行的检查请明确说明。

## 实验文件

- 原始数据和预处理图片保留在本地，目录约定见 [data/README.md](data/README.md)。
- `outputs/` 中已提交的模型、JSON 与图表是历史实验材料，请勿把本地试跑结果覆盖进来。
- 新实验默认可放到已忽略的 `outputs/local-run/`；需要更新公开结果时，同时说明生成命令与指标来源。
- 文档中的命令应与脚本实际参数一致，变更后检查相关 README 和 Notebook 说明。
