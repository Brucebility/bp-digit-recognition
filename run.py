# run_experiment.py - 一键运行整个实验
import subprocess
import sys
import os


def install_requirements():
    """安装依赖包"""
    print("=" * 60)
    print("安装依赖包")
    print("=" * 60)

    requirements = ['numpy', 'matplotlib', 'Pillow']

    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装成功")
        except:
            print(f"✗ {package} 安装失败，尝试继续...")

    print()


def setup_folders():
    """创建必要的文件夹"""
    print("=" * 60)
    print("创建项目文件夹")
    print("=" * 60)

    folders = ['models', 'test_images']

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✓ 创建文件夹: {folder}")
        else:
            print(f"✓ 文件夹已存在: {folder}")

    # 检查MNIST数据
    mnist_files = ['data/mnist/train-images-idx3-ubyte',
                   'data/mnist/train-labels-idx1-ubyte',
                   'data/mnist/t10k-images-idx3-ubyte',
                   'data/mnist/t10k-labels-idx1-ubyte']

    print("\n检查MNIST数据文件:")
    all_exist = True
    for file in mnist_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (未找到)")
            all_exist = False

    if not all_exist:
        print("\n警告：缺少MNIST数据文件！")
        print("请将MNIST文件放入 data/mnist/ 目录")
        print("需要的文件：")
        print("  - train-images-idx3-ubyte")
        print("  - train-labels-idx1-ubyte")
        print("  - t10k-images-idx3-ubyte")
        print("  - t10k-labels-idx1-ubyte")
        print("\n可以从这里下载：http://yann.lecun.com/exdb/mnist/")

    print()


def run_experiment():
    """运行实验"""
    print("=" * 60)
    print("BP神经网络手写数字识别实验")
    print("=" * 60)

    while True:
        print("\n请选择要执行的步骤:")
        print("1. 训练模型 (运行 train.py)")
        print("2. 测试模型 (运行 test.py)")
        print("3. 测试自制手写数字 (运行 custom_test.py)")
        print("4. 完整实验 (1→2→3)")
        print("5. 退出")

        choice = input("\n请输入选择 (1-5): ").strip()

        if choice == '1':
            print("\n开始训练模型...")
            from digit import train
            train.main()

        elif choice == '2':
            print("\n开始测试模型...")
            from digit import test
            test.test_model()

        elif choice == '3':
            print("\n开始测试自制手写数字...")
            import custom_test
            custom_test.test_custom_images()

        elif choice == '4':
            print("\n开始完整实验...")
            print("\n步骤1: 训练模型")
            print("-" * 30)
            from digit import train
            model = train.main()

            input("\n按回车键继续步骤2: 测试模型...")
            print("\n步骤2: 测试模型")
            print("-" * 30)
            from digit import test
            test.test_model()

            input("\n按回车键继续步骤3: 测试自制手写数字...")
            print("\n步骤3: 测试自制手写数字")
            print("-" * 30)
            print("请先在 test_images 文件夹中放入手写数字图片")
            print("(0.png, 1.png, ..., 9.png)")
            input("准备好后按回车键继续...")
            import custom_test
            custom_test.test_custom_images()

            print("\n" + "=" * 60)
            print("完整实验完成！")
            print("=" * 60)

        elif choice == '5':
            print("退出实验")
            break

        else:
            print("无效选择，请重新输入")

        print()


if __name__ == "__main__":
    # 安装依赖
    install_requirements()

    # 设置文件夹
    setup_folders()

    # 运行实验
    run_experiment()