import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
data = pd.read_excel('客运量总计.xlsx')

# 提取时间序列和值
# 假设时间序列是第一列，值是第二列
time = data.iloc[:, 0]
values = data.iloc[:, 1]

# 绘制曲线图
plt.figure(figsize=(10, 6))  # 设置图形的大小
plt.plot(time, values, marker='o', linestyle='-', color='r')  # 绘制曲线图，蓝色实线，带圆点标记

# 添加标题和标签
plt.xlabel('时间', fontdict={'size': 14})  # x轴标签
plt.ylabel('客运量总计', fontdict={'size': 14})  # y轴标签

# 添加网格线
plt.grid(True)

# 优化 x 轴和 y 轴的显示
plt.xticks(rotation=45)  # 旋转 x 轴标签，便于阅读
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

# 显示图形
plt.show()