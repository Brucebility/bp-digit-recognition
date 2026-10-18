# test.py - 测试MNIST数据集
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'
from digit.network import NeuralNetwork
from digit.utils import load_mnist, one_hot_encode


def test_model():
    print("=" * 60)
    print("BP神经网络手写数字识别 - 模型测试")
    print("=" * 60)

    # 1. 加载测试数据
    print("\n[1] 加载测试数据...")
    _, _, test_images, test_labels = load_mnist()
    test_labels_onehot = one_hot_encode(test_labels)

    # 2. 加载模型
    print("\n[2] 加载训练好的模型...")
    model = NeuralNetwork(784, 256, 10)

    try:
        model.load_weights(str(MODEL_DIR / 'bp_model.npz'))
        print("模型加载成功！")
    except:
        print("错误：找不到模型文件，请先运行 train.py")
        return

    # 3. 测试模型
    print("\n[3] 测试模型性能...")
    test_acc = model.accuracy(test_images, test_labels_onehot)
    print(f"测试集准确率: {test_acc:.2%}")

    # 4. 详细分析
    print("\n[4] 详细分析...")
    analyze_model_performance(model, test_images, test_labels)

    # 5. 可视化错误案例
    print("\n[5] 可视化错误案例...")
    visualize_errors(model, test_images, test_labels, 20)

    return model


def analyze_model_performance(model, test_images, test_labels):
    """分析模型性能"""
    predictions = model.predict(test_images)

    # 计算各类别的准确率
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    for i in range(len(test_labels)):
        true_label = test_labels[i]
        pred_label = predictions[i]
        class_total[true_label] += 1
        if true_label == pred_label:
            class_correct[true_label] += 1

    print("\n各类别准确率:")
    print("-" * 30)
    for i in range(10):
        accuracy = class_correct[i] / class_total[i] * 100
        print(f"数字 {i}: {accuracy:5.1f}% ({int(class_correct[i])}/{int(class_total[i])})")

    # 混淆矩阵（简化版）
    print("\n常见错误分类:")
    print("-" * 30)
    errors = {}
    for i in range(len(test_labels)):
        true_label = test_labels[i]
        pred_label = predictions[i]
        if true_label != pred_label:
            key = (true_label, pred_label)
            errors[key] = errors.get(key, 0) + 1

    # 显示最多的5个错误
    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:5]
    for (true, pred), count in sorted_errors:
        print(f"{true} → {pred}: {count}次")


def visualize_errors(model, test_images, test_labels, num_errors=20):
    """可视化错误分类的案例"""
    predictions = model.predict(test_images)

    # 找出错误分类的索引
    error_indices = []
    for i in range(len(test_labels)):
        if test_labels[i] != predictions[i]:
            error_indices.append(i)
        if len(error_indices) >= num_errors:
            break

    if not error_indices:
        print("没有找到错误分类！")
        return

    print(f"\n显示 {len(error_indices)} 个错误案例:")

    # 绘制错误案例
    cols = 5
    rows = (len(error_indices) + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))
    for idx, error_idx in enumerate(error_indices):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(test_images[error_idx].reshape(28, 28), cmap='gray')

        true_label = test_labels[error_idx]
        pred_label = predictions[error_idx]

        plt.title(f'True: {true_label} → Pred: {pred_label}',
                  color='red', fontsize=10)
        plt.axis('off')

    plt.suptitle('Error Cases in Test Set', fontsize=14)
    plt.tight_layout()

    # 保存到上级目录的models文件夹
    os.makedirs('../models', exist_ok=True)
    plt.savefig(str(MODEL_DIR / 'error_cases.png'), dpi=150)
    print(f"错误案例图已保存到: {MODEL_DIR / 'error_cases.png'}")
    plt.show()


if __name__ == "__main__":
    model = test_model()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("查看生成的文件:")
    print("  - ../models/error_cases.png: 错误分类案例")
    print("=" * 60)