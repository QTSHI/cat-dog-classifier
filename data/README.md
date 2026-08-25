# 数据目录

数据集图片不会提交到 GitHub，请按下面的结构放置：

```text
data/
├── raw/
│   ├── train/cats
│   ├── train/dogs
│   ├── test/cats
│   └── test/dogs
└── processed/
    ├── train/cats
    ├── train/dogs
    ├── test/cats
    └── test/dogs
```

`raw` 保存原图，`processed` 保存裁剪为 $128\times128$ 的图片。
