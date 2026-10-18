# MNIST 数据目录

本目录需要放置 MNIST 原始数据（**不随仓库提交**，共约 46 MB）。

## 需要 4 个文件

```
data/mnist/
├── train-images-idx3-ubyte     # 60000 张训练图片（约 45 MB）
├── train-labels-idx1-ubyte     # 60000 个训练标签
├── t10k-images-idx3-ubyte      # 10000 张测试图片
└── t10k-labels-idx1-ubyte      # 10000 个测试标签
```

## 下载

官方地址：http://yann.lecun.com/exdb/mnist/

下载后解压（文件本身无扩展名，保持原文件名），放入本目录即可。脚本 `digit/utils.py` 会自动从这里读取。

## 说明

`models/bp_model.npz` 已附带训练好的权重，只想验证识别效果时不需要这个目录，跑 `python custom_test.py` 即可。
