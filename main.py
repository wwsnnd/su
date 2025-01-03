import cv2
import numpy as np
import os

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

def main():
    # 图像和 Ground Truth 掩码路径
    image_path = r"G:\xiaosu\detection\train\images\20150921_131833_image471.png"  # 替换为图像路径
    mask_path = r"G:\xiaosu\detection\train\masks\20150921_131833_image471.png"    # 替换为 Ground Truth 掩码路径

    # 加载图像和 Ground Truth
    image = cv2.imread(image_path)
    gt_contours = load_ground_truth(mask_path)

    # 检测苹果
    detected_contours = detect_apples(image)

    # 评估检测结果
    precision, recall, f1_score = evaluate_detection(detected_contours, gt_contours)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

    # 可视化结果
    for cnt in detected_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Detected Apples", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()