# custom_test.py - 修正导入版本
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

from digit.network import NeuralNetwork


def preprocess_image(img_path):
    """预处理手写数字图片"""
    try:
        # 1. 打开图片并转为灰度
        img = Image.open(img_path).convert('L')

        # 2. 调整大小为28×28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # 3. 转为numpy数组
        img_array = np.array(img)

        # 4. 归一化并反转（MNIST是黑底白字）
        img_array = 1.0 - img_array / 255.0

        # 5. 展平
        img_flat = img_array.flatten()

        return img_flat

    except Exception as e:
        print(f"预处理图片失败 {img_path}: {e}")
        return None


def main():
    print("=" * 60)
    print("自制手写数字图片识别测试")
    print("=" * 60)

    # 1. 加载训练好的模型
    print("\n[1] 加载训练好的BP神经网络模型...")
    model = NeuralNetwork(784, 256, 10)

    try:
        model.load_weights('models/bp_model.npz')
        print("模型加载成功！")
    except FileNotFoundError:
        print("错误：找不到模型文件 'models/bp_model.npz'")
        print("请先运行 train.py 训练模型")
        return

    # 2. 检查test_images文件夹
    test_folder = 'test_images'
    if not os.path.exists(test_folder):
        print(f"\n错误：找不到文件夹 '{test_folder}'")
        print("请创建 test_images 文件夹，并放入手写数字图片：")
        print("0.png, 1.png, ..., 9.png")
        return

    # 3. 测试所有0-9.png
    print(f"\n[2] 测试 {test_folder} 文件夹中的手写数字...")
    print("-" * 60)

    results = []
    images_found = []

    for i in range(10):
        img_path = os.path.join(test_folder, f'{i}.png')

        if not os.path.exists(img_path):
            print(f"警告：图片不存在: {i}.png")
            continue

        images_found.append(i)

        # 预处理图片
        processed_img = preprocess_image(img_path)
        if processed_img is None:
            continue

        # 预测
        prob = model.forward(processed_img.reshape(1, -1))
        prediction = np.argmax(prob[0])
        confidence = np.max(prob[0]) * 100

        # 记录结果
        correct = (prediction == i)
        results.append({
            'digit': i,
            'prediction': prediction,
            'confidence': confidence,
            'correct': correct
        })

        # 显示结果
        status = "✓" if correct else "✗"
        color = "✓" if correct else "✗"

        print(f"数字 {i}: 预测 = {prediction}, 置信度 = {confidence:.1f}%, 结果 = {color}")

    if not results:
        print("没有找到任何测试图片！")
        return

    # 4. 统计结果
    print("\n[3] 测试结果统计")
    print("-" * 60)

    total = len(results)
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / total * 100

    print(f"找到图片数: {total}/10")
    print(f"正确识别数: {correct_count}")
    print(f"识别准确率: {accuracy:.1f}%")

    # 5. 分析错误案例
    print("\n[4] 错误分析")
    print("-" * 60)

    wrong_cases = [r for r in results if not r['correct']]
    if wrong_cases:
        print(f"共有 {len(wrong_cases)} 个识别错误:")
        for r in wrong_cases:
            print(f"  数字 {r['digit']} → 误识别为 {r['prediction']} (置信度: {r['confidence']:.1f}%)")
    else:
        print("所有数字识别正确！")

    # 6. 简单可视化
    if results:
        print("\n[5] 显示测试图片...")
        show_simple_results(results, test_folder)

    return results


def show_simple_results(results, test_folder):
    """简单显示结果"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i in range(10):
        row = i // 5
        col = i % 5
        ax = axes[row, col]

        img_path = os.path.join(test_folder, f'{i}.png')

        if os.path.exists(img_path):
            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28))
            ax.imshow(img, cmap='gray')

            # 查找结果
            result = next((r for r in results if r['digit'] == i), None)
            if result:
                title_color = 'green' if result['correct'] else 'red'
                title = f'True:{i}→Pred:{result["prediction"]}'
                ax.set_title(title, color=title_color, fontsize=9)
        else:
            ax.text(0.5, 0.5, f'{i}.png\n不存在',
                    ha='center', va='center', fontsize=9)
            ax.set_title(f'数字 {i}', fontsize=9)

        ax.axis('off')

    plt.suptitle('手写数字识别结果', fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 检查test_images文件夹
    if not os.path.exists('test_images'):
        print("创建 test_images 文件夹...")
        os.makedirs('test_images')
        print("请在 test_images 文件夹中放入手写数字图片：")
        print("0.png, 1.png, ..., 9.png")
        print("然后重新运行本程序")
    else:
        # 运行测试
        test_results = main()

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)