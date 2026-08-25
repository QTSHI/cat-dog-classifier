# 数据放置方式

图片没有上传到仓库，本地按下面的目录放置：

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

`raw` 放原图，`processed` 放裁剪为 $128\times128$ 的图片。
