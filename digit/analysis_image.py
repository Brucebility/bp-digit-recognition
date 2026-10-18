# 生成十张图片_强制宋体.py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm

print("正在生成十张手写数字分析图片...")

# ========== 强制设置宋体 ==========
# 方法：添加字体文件或使用系统宋体
try:
    # 方法1：直接使用SimSun（宋体）
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    # 验证字体是否可用
    fig_test = plt.figure()
    ax_test = fig_test.add_subplot(111)
    ax_test.text(0.5, 0.5, "测试宋体", fontsize=12)
    plt.close(fig_test)
    print("✓ 宋体设置成功")

except Exception as e:
    print(f"⚠ 宋体设置失败: {e}")
    print("将使用默认字体")

# ========== 数据定义 ==========
# 这里使用你提供的部分真实数据和模拟数据
数据列表 = [
    # 数字 0 - 实际数据
    {
        '文件名': '0.png',
        '真实标签': 0,
        '预测标签': 2,
        '置信度': 59.9,
        '是否正确': False,
        '概率分布': [0.3, 0.0, 95.1, 0.2, 0.1, 0.0, 0.1, 3.1, 0.2, 0.9]
    },
    # 数字 1 - 模拟数据
    {
        '文件名': '1.png',
        '真实标签': 1,
        '预测标签': 1,
        '置信度': 45.4,
        '是否正确': True,
        '概率分布': [2.0, 45.4, 10.0, 8.0, 5.0, 6.0, 7.0, 8.0, 4.0, 4.6]
    },
    # 数字 2 - 模拟数据
    {
        '文件名': '2.png',
        '真实标签': 2,
        '预测标签': 2,
        '置信度': 45.2,
        '是否正确': True,
        '概率分布': [3.0, 8.0, 45.2, 10.0, 6.0, 5.0, 8.0, 7.0, 4.0, 3.8]
    },
    # 数字 3 - 模拟数据
    {
        '文件名': '3.png',
        '真实标签': 3,
        '预测标签': 3,
        '置信度': 67.9,
        '是否正确': True,
        '概率分布': [2.0, 5.0, 8.0, 67.9, 3.0, 4.0, 2.0, 3.0, 2.0, 3.1]
    },
    # 数字 4 - 实际数据
    {
        '文件名': '4.png',
        '真实标签': 4,
        '预测标签': 4,
        '置信度': 44.4,
        '是否正确': True,
        '概率分布': [0.0, 0.8, 1.0, 0.3, 44.4, 0.2, 19.5, 0.1, 22.4, 11.2]
    },
    # 数字 5 - 模拟数据
    {
        '文件名': '5.png',
        '真实标签': 5,
        '预测标签': 5,
        '置信度': 43.0,
        '是否正确': True,
        '概率分布': [3.0, 5.0, 10.0, 15.0, 8.0, 43.0, 6.0, 5.0, 3.0, 2.0]
    },
    # 数字 6 - 模拟数据
    {
        '文件名': '6.png',
        '真实标签': 6,
        '预测标签': 5,
        '置信度': 66.3,
        '是否正确': False,
        '概率分布': [2.0, 3.0, 8.0, 5.0, 4.0, 66.3, 5.0, 2.0, 2.0, 2.7]
    },
    # 数字 7 - 实际数据
    {
        '文件名': '7.png',
        '真实标签': 7,
        '预测标签': 3,
        '置信度': 55.1,
        '是否正确': False,
        '概率分布': [0.0, 0.8, 21.2, 55.1, 0.2, 0.0, 0.1, 22.4, 0.2, 0.1]
    },
    # 数字 8 - 模拟数据
    {
        '文件名': '8.png',
        '真实标签': 8,
        '预测标签': 6,
        '置信度': 74.5,
        '是否正确': False,
        '概率分布': [3.0, 2.0, 4.0, 3.0, 2.0, 3.0, 74.5, 2.0, 3.0, 3.5]
    },
    # 数字 9 - 模拟数据
    {
        '文件名': '9.png',
        '真实标签': 9,
        '预测标签': 3,
        '置信度': 36.6,
        '是否正确': False,
        '概率分布': [2.0, 5.0, 10.0, 36.6, 12.0, 8.0, 7.0, 6.0, 8.0, 5.4]
    }
]


def 创建处理图像(数字):
    """创建28x28的处理后图像"""
    图像 = np.zeros((28, 28))

    if 数字 == 0:
        # 画圆
        for i in range(28):
            for j in range(28):
                dist = np.sqrt((i - 14) ** 2 + (j - 14) ** 2)
                if 6 <= dist <= 9:
                    图像[i, j] = 1.0
    elif 数字 == 1:
        # 画竖线
        for i in range(8, 20):
            for j in range(12, 16):
                图像[i, j] = 1.0
    elif 数字 == 2:
        # 画2
        for i in range(6, 9):
            for j in range(6, 22):
                图像[i, j] = 0.8
        for i in range(12, 15):
            for j in range(6, 22):
                图像[i, j] = 0.8
        for i in range(18, 21):
            for j in range(6, 22):
                图像[i, j] = 0.8
    elif 数字 == 3:
        # 画3
        for i in range(6, 21):
            for j in range(6, 9):
                图像[i, j] = 0.8
            for j in range(19, 22):
                图像[i, j] = 0.8
        for i in range(6, 9):
            for j in range(6, 22):
                图像[i, j] = 0.8
        for i in range(12, 15):
            for j in range(6, 22):
                图像[i, j] = 0.8
        for i in range(18, 21):
            for j in range(6, 22):
                图像[i, j] = 0.8
    elif 数字 == 4:
        # 画4
        for i in range(6, 21):
            for j in range(15, 18):
                图像[i, j] = 0.8
        for j in range(6, 18):
            for i in range(12, 15):
                图像[i, j] = 0.8
    elif 数字 == 5:
        # 画5
        for i in range(6, 9):
            for j in range(6, 22):
                图像[i, j] = 0.8
        for i in range(12, 15):
            for j in range(6, 22):
                图像[i, j] = 0.8
        for i in range(18, 21):
            for j in range(6, 22):
                图像[i, j] = 0.8
    elif 数字 == 6:
        # 画6
        for i in range(6, 21):
            for j in range(6, 9):
                图像[i, j] = 0.8
        for j in range(6, 18):
            for i in range(18, 21):
                图像[i, j] = 0.8
        for i in range(12, 15):
            for j in range(6, 18):
                图像[i, j] = 0.8
    elif 数字 == 7:
        # 画7
        for i in range(6, 9):
            for j in range(6, 22):
                图像[i, j] = 0.8
        for i in range(6, 21):
            for j in range(19, 22):
                图像[i, j] = 0.8
    elif 数字 == 8:
        # 画8
        for i in range(6, 21):
            for j in range(6, 9):
                图像[i, j] = 0.8
            for j in range(19, 22):
                图像[i, j] = 0.8
        for i in range(6, 9):
            for j in range(6, 22):
                图像[i, j] = 0.8
        for i in range(12, 15):
            for j in range(6, 22):
                图像[i, j] = 0.8
        for i in range(18, 21):
            for j in range(6, 22):
                图像[i, j] = 0.8
    elif 数字 == 9:
        # 画9
        for i in range(6, 15):
            for j in range(6, 9):
                图像[i, j] = 0.8
        for j in range(6, 18):
            for i in range(12, 15):
                图像[i, j] = 0.8
        for i in range(6, 9):
            for j in range(6, 18):
                图像[i, j] = 0.8
        for j in range(15, 18):
            for i in range(6, 21):
                图像[i, j] = 0.8

    return 图像


def 生成图片(索引, 数据):
    """生成单张分析图片"""

    文件名 = 数据['文件名']
    真实标签 = 数据['真实标签']
    预测标签 = 数据['预测标签']
    置信度 = 数据['置信度']
    是否正确 = 数据['是否正确']
    概率分布 = 数据['概率分布']

    # 创建画布
    fig = plt.figure(figsize=(12, 10))

    # 1. 左上：测试信息
    ax1 = plt.subplot(2, 2, 1)
    ax1.axis('off')

    # 使用更简单的文本显示
    文本内容 = f"测试图片：{文件名}\n\n"
    文本内容 += f"图片：{文件名}\n"
    文本内容 += f"真实：{真实标签}\n"
    文本内容 += f"预测：{预测标签}\n"
    文本内容 += f"置信度：{置信度:.1f}%\n"

    # 使用默认字体
    ax1.text(0.1, 0.5, 文本内容, fontsize=13,
             verticalalignment='center', linespacing=1.8,
             family='sans-serif')

    # 2. 右上：处理后图像
    ax2 = plt.subplot(2, 2, 2)
    处理图像 = 创建处理图像(真实标签)
    ax2.imshow(处理图像, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('处理后图像', fontsize=12, pad=10, family='sans-serif')
    ax2.axis('off')

    # 3. 左下：预测概率分布
    ax3 = plt.subplot(2, 2, 3)

    数字列表 = list(range(10))

    # 创建柱状图
    柱状图 = ax3.bar(数字列表, 概率分布, color='lightblue',
                     edgecolor='black', alpha=0.8, width=0.7)

    # 高亮预测标签
    柱状图[预测标签].set_color('orange')
    if 预测标签 != 真实标签:
        柱状图[真实标签].set_color('lightgreen')

    # 使用英文标签避免乱码
    ax3.set_xlabel('Digit', fontsize=12)
    ax3.set_ylabel('Probability (%)', fontsize=12)
    ax3.set_title('Probability Distribution', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(数字列表)

    # 设置Y轴范围
    最大概率 = max(概率分布)
    ax3.set_ylim([0, 最大概率 * 1.15])

    # 添加网格
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 添加概率值
    for 柱, 概率值 in zip(柱状图, 概率分布):
        if 概率值 > 1:
            ax3.text(柱.get_x() + 柱.get_width() / 2., 柱.get_height() + 最大概率 * 0.02,
                     f'{概率值:.1f}%', ha='center', va='bottom', fontsize=9)

    # 4. 右下：详细信息
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # 使用等宽字体显示表格
    详细信息 = f"预测结果：{预测标签}\n"
    详细信息 += f"置信度：{置信度:.1f}%\n"
    详细信息 += f"是否正确：{'是' if 是否正确 else '否'}\n\n"
    详细信息 += "各数字概率：\n"

    for i, 概率值 in enumerate(概率分布):
        详细信息 += f"数字 {i}： {概率值:.1f}%\n"

    # 使用等宽字体避免对齐问题
    ax4.text(0.1, 0.95, 详细信息, fontsize=11,
             verticalalignment='top', family='monospace',
             linespacing=1.6)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    输出文件名 = f"{真实标签}_分析.png"
    plt.savefig(输出文件名, dpi=150, bbox_inches='tight')
    plt.close()

    return 输出文件名


# ========== 主程序 ==========
def 主程序():
    print("开始生成分析图片...")
    print("=" * 50)

    成功数量 = 0

    for i, 数据 in enumerate(数据列表):
        try:
            输出文件 = 生成图片(i, 数据)
            成功数量 += 1

            状态 = "✓" if 数据['是否正确'] else "✗"
            print(f"{状态} 数字 {数据['真实标签']}: {输出文件}")

        except Exception as e:
            print(f"✗ 生成数字 {数据['真实标签']} 时出错: {str(e)[:50]}")

    print("=" * 50)
    print(f"✅ 生成完成！")
    print(f"成功生成: {成功数量}/10 张图片")

    # 显示生成的图片
    print("\n📁 生成的图片文件:")
    print("-" * 40)
    for file in os.listdir('.'):
        if file.endswith('_分析.png'):
            文件大小 = os.path.getsize(file) / 1024
            print(f"  {file} ({文件大小:.1f} KB)")


# 运行程序
if __name__ == "__main__":
    主程序()