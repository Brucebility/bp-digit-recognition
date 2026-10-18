# utils.py - 工具函数
import numpy as np
import struct
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'mnist'


def load_mnist(data_dir=None):
    """
    加载MNIST数据
    注意：路径是 ../data/mnist（因为utils.py在digit文件夹里）
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    print("加载MNIST数据集...")

    try:
        # 加载训练数据
        with open(data_dir / 'train-images-idx3-ubyte', 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            train_images = np.frombuffer(f.read(), dtype=np.uint8)
            train_images = train_images.reshape(num, rows * cols) / 255.0

        with open(data_dir / 'train-labels-idx1-ubyte', 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            train_labels = np.frombuffer(f.read(), dtype=np.uint8)

        # 加载测试数据
        with open(data_dir / 't10k-images-idx3-ubyte', 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            test_images = np.frombuffer(f.read(), dtype=np.uint8)
            test_images = test_images.reshape(num, rows * cols) / 255.0

        with open(data_dir / 't10k-labels-idx1-ubyte', 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            test_labels = np.frombuffer(f.read(), dtype=np.uint8)

        print(f"训练集: {train_images.shape[0]} 样本")
        print(f"测试集: {test_images.shape[0]} 样本")

        return train_images, train_labels, test_images, test_labels

    except FileNotFoundError as e:
        print(f"错误：找不到MNIST文件: {e}")
        print("请确保MNIST文件在 ../data/mnist/ 目录下")
        print("需要的文件：")
        print("  - train-images-idx3-ubyte")
        print("  - train-labels-idx1-ubyte")
        print("  - t10k-images-idx3-ubyte")
        print("  - t10k-labels-idx1-ubyte")
        raise


def one_hot_encode(labels, num_classes=10):
    """One-hot编码"""
    n = len(labels)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), labels] = 1
    return one_hot


def train_val_split(images, labels, val_ratio=0.2):
    """划分训练集和验证集"""
    n = len(images)
    val_size = int(n * val_ratio)
    indices = np.random.permutation(n)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (images[train_idx], labels[train_idx],
            images[val_idx], labels[val_idx])