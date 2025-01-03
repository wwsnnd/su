import cv2
import numpy as np
import os
import csv


def preprocess_image(image):
    """
    图像预处理：灰度化、高斯滤波、对比度增强
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return enhanced


def detect_apples(image):
    """
    检测苹果：颜色阈值分割 + 形态学操作
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def load_ground_truth(mask_path):
    """
    加载 Ground Truth 掩码图像并解析苹果区域
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比 (IoU)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union != 0 else 0


def evaluate_detection(detected_contours, gt_contours, iou_threshold=0.5):
    """
    评估检测结果：计算精确率、召回率和 F1 分数
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # 将检测到的轮廓转换为边界框
    detected_boxes = [cv2.boundingRect(cnt) for cnt in detected_contours]
    detected_boxes = [(x, y, x + w, y + h) for (x, y, w, h) in detected_boxes]

    # 将 Ground Truth 轮廓转换为边界框
    gt_boxes = [cv2.boundingRect(cnt) for cnt in gt_contours]
    gt_boxes = [(x, y, x + w, y + h) for (x, y, w, h) in gt_boxes]

    # 匹配检测框和 Ground Truth 框
    matched_gt = set()
    for det_box in detected_boxes:
        matched = False
        for i, gt_box in enumerate(gt_boxes):
            if i not in matched_gt and calculate_iou(det_box, gt_box) >= iou_threshold:
                true_positives += 1
                matched_gt.add(i)
                matched = True
                break
        if not matched:
            false_positives += 1

    false_negatives = len(gt_boxes) - len(matched_gt)

    # 计算精确率、召回率和 F1 分数
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def process_dataset(image_folder, mask_folder, output_folder):
    """
    处理数据集中的所有图像，并保存结果
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建 CSV 文件保存结果
    csv_path = os.path.join(output_folder, "results.csv")
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image", "Apple Count", "Precision", "Recall", "F1 Score"])

        # 遍历图像文件夹
        for filename in os.listdir(image_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # 加载图像和 Ground Truth
                image_path = os.path.join(image_folder, filename)
                mask_path = os.path.join(mask_folder, filename.replace(".jpg", ".png").replace(".JPG", ".png"))

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Unable to load image at {image_path}")
                    continue

                gt_contours = load_ground_truth(mask_path)
                if not gt_contours:
                    print(f"Warning: No ground truth contours found for {filename}.")
                    continue

                # 检测苹果
                detected_contours = detect_apples(image)

                # 评估检测结果
                precision, recall, f1_score = evaluate_detection(detected_contours, gt_contours)

                # 计数苹果
                apple_count = len(detected_contours)

                # 保存结果到 CSV
                writer.writerow([filename, apple_count, precision, recall, f1_score])

                # 可视化结果并保存到输出文件夹
                output_image = image.copy()
                for cnt in detected_contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                output_image_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_image_path, output_image)

                print(
                    f"Processed {filename}: Apple Count = {apple_count}, Precision = {precision:.2f}, Recall = {recall:.2f}, F1 Score = {f1_score:.2f}")


if __name__ == "__main__":
    # 数据集路径
    image_folder = r"G:\xiaosu\test_data\segmentation\images"  # 替换为图像文件夹路径
    mask_folder = r"G:\xiaosu\test_data\segmentation\masks" # 替换为 Ground Truth 掩码文件夹路径
    output_folder = r"G:\xiaosu\1"  # 替换为输出文件夹路径

    # 处理数据集
    process_dataset(image_folder, mask_folder, output_folder)