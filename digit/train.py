# train.py - 训练BP神经网络
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from digit.utils import load_mnist, one_hot_encode, train_val_split
from digit.network import NeuralNetwork


def main():
    print("=" * 60)
    print("BP神经网络训练")
    print("=" * 60)

    # 1. 加载数据
    train_images, train_labels, test_images, test_labels = load_mnist()

    # 2. 数据预处理
    train_images, train_labels, val_images, val_labels = train_val_split(
        train_images, train_labels, val_ratio=0.2
    )

    # One-hot编码
    train_labels_onehot = one_hot_encode(train_labels)
    val_labels_onehot = one_hot_encode(val_labels)
    test_labels_onehot = one_hot_encode(test_labels)

    print(f"\n数据划分:")
    print(f"  训练集: {train_images.shape[0]} 样本")
    print(f"  验证集: {val_images.shape[0]} 样本")
    print(f"  测试集: {test_images.shape[0]} 样本")

    # 3. 创建神经网络
    input_size = 784  # 28x28
    hidden_size = 256
    output_size = 10

    model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.01)
    print(f"\n网络结构: {input_size}-{hidden_size}-{output_size}")
    print(f"学习率: 0.01")

    # 4. 训练参数
    epochs = 15
    batch_size = 128

    # 记录训练过程
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\n开始训练...")
    print("-" * 60)

    # 5. 训练循环
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        # 随机打乱
        indices = np.random.permutation(len(train_images))
        X_shuffled = train_images[indices]
        y_shuffled = train_labels_onehot[indices]

        # 批量训练
        for i in range(0, len(train_images), batch_size):
            batch_X = X_shuffled[i:i + batch_size]
            batch_y = y_shuffled[i:i + batch_size]

            # 前向传播
            y_pred = model.forward(batch_X)

            # 计算损失
            loss = model.compute_loss(y_pred, batch_y)
            epoch_loss += loss

            # 反向传播
            model.backward(batch_X, batch_y)
            num_batches += 1

        # 平均损失
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # 计算准确率
        train_acc = model.accuracy(train_images[:5000], train_labels_onehot[:5000])
        val_acc = model.accuracy(val_images, val_labels_onehot)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2%} | "
              f"Val Acc: {val_acc:.2%}")

    training_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {training_time:.2f}秒")

    # 6. 测试集评估
    test_acc = model.accuracy(test_images, test_labels_onehot)
    print(f"\n测试集准确率: {test_acc:.2%}")

    # 7. 保存模型
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(MODEL_DIR / 'bp_model.npz'))
    print(f"模型已保存到: {MODEL_DIR / 'bp_model.npz'}")

    # 8. 可视化
    visualize_training(train_losses, train_accuracies, val_accuracies, test_acc)

    return model


def visualize_training(train_losses, train_accuracies, val_accuracies, test_acc):
    """可视化训练过程"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, train_accuracies, 'g-', linewidth=2, label='Train')
    ax2.plot(epochs, val_accuracies, 'r-', linewidth=2, label='Validation')
    ax2.axhline(y=test_acc, color='orange', linestyle='--', label=f'Test: {test_acc:.2%}')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('BP Neural Network Training Results', fontsize=14)
    plt.tight_layout()

    # 保存图像
    plt.savefig(str(MODEL_DIR / 'training_results.png'), dpi=150)
    print(f"训练结果图已保存到: {MODEL_DIR / 'training_results.png'}")
    plt.show()


if __name__ == "__main__":
    main()