import cv2
import numpy as np

def draw_bbox(img, xmin, ymin, xmax, ymax, label, color=(0, 255, 0), thickness=2):
    """绘制检测框"""
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    # 添加标签
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_ymin = max(ymin, label_size[1] + 10)
    cv2.rectangle(img, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin), color, -1)
    cv2.putText(img, label, (xmin, label_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img
