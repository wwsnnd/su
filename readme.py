苹果检测与计数项目
项目概述
本项目旨在通过传统的图像处理方法检测和计数果园中的苹果。项目使用公开的 MinneApple 数据集，通过颜色阈值分割、形态学操作和轮廓检测等技术实现苹果的检测与计数。每张图像的检测结果会保存到输出文件夹，并生成一个包含每张图像苹果数量、精确率、召回率和 F1 分数的 CSV 文件。
项目结构
project/
│
├── images/                  # 存放输入图像
├── masks/                   # 存放 Ground Truth 掩码图像
├── output/                  # 存放输出图像和结果文件
│   ├── results.csv          # 每张图像的检测结果
│   └── processed_images/    # 处理后的图像（带检测框）
├── main.py                  # 主程序
└── README.md                # 项目说明文件
环境要求
Python 3.7 或更高版本

OpenCV (cv2)

NumPy (numpy)

安装依赖
运行以下命令安装所需的 Python 库：-
pip install opencv-python numpy
数据集
本项目使用 MinneApple 数据集，包含苹果图像及其对应的 Ground Truth 掩码图像。数据集可以从以下链接下载：

MinneApple Dataset

数据集结构
images/：存放苹果图像（.jpg 或 .png 格式）。

masks/：存放 Ground Truth 掩码图像（.png 格式），文件名与图像文件对应。
使用方法
1. 准备数据集
将苹果图像放入 images/ 文件夹。

将 Ground Truth 掩码图像放入 masks/ 文件夹。

2. 运行程序
在终端中运行以下命令：
python 1.py
3. 查看结果
处理后的图像会保存到 output/processed_images/ 文件夹。

检测结果（包括苹果数量、精确率、召回率和 F1 分数）会保存到 output/results.csv 文件中。
代码说明
主要函数
preprocess_image(image)：

对图像进行预处理，包括灰度化、高斯滤波和对比度增强。

detect_apples(image)：

使用颜色阈值分割和形态学操作检测苹果，并返回检测到的轮廓。

load_ground_truth(mask_path)：

加载 Ground Truth 掩码图像，并解析苹果区域。

calculate_iou(box1, box2)：

计算两个边界框的交并比（IoU）。

evaluate_detection(detected_contours, gt_contours)：

评估检测结果，计算精确率、召回率和 F1 分数。

process_dataset(image_folder, mask_folder, output_folder)：

处理数据集中的所有图像，保存结果并生成 CSV 文件。
输出结果
1. 处理后的图像
每张图像会标注检测到的苹果区域，并保存到 output/processed_images/ 文件夹。

2. CSV 文件
output/results.csv 文件包含以下字段：

Image：图像文件名。

Apple Count：检测到的苹果数量。

Precision：检测的精确率。

Recall：检测的召回率。

F1 Score：检测的 F1 分数。