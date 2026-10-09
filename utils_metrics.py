import torch
import numpy as np
from tqdm import tqdm
from thop import profile

def calculate_iou(box1, box2):
    """计算IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def calculate_map(model, dataloader, device, iou_thres=0.5):
    """计算mAP"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, boxes in tqdm(dataloader, desc='Calculating mAP'):
            imgs = imgs.to(device)
            outputs = model(imgs)

            # 解析预测框和真实框
            for i in range(len(imgs)):
                pred_boxes = outputs[i].cpu().numpy() if len(outputs) > i else []
                target_boxes = boxes[i].cpu().numpy() if len(boxes) > i else []

                all_preds.append(pred_boxes)
                all_targets.append(target_boxes)

    # 计算mAP（简化实现，可替换为COCO官方实现）
    aps = []
    for preds, targets in zip(all_preds, all_targets):
        if len(targets) == 0:
            continue
        if len(preds) == 0:
            aps.append(0)
            continue

        # 按置信度排序
        preds = sorted(preds, key=lambda x: x[4], reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        target_matched = np.zeros(len(targets))

        for i, pred in enumerate(preds):
            best_iou = 0
            best_idx = -1
            for j, target in enumerate(targets):
                if target_matched[j]:
                    continue
                iou = calculate_iou(pred[:4], target[:4])
                if iou > best_iou and iou >= iou_thres:
                    best_iou = iou
                    best_idx = j

            if best_idx != -1:
                target_matched[best_idx] = 1
                tp[i] = 1
            else:
                fp[i] = 1

        # 计算AP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(targets)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        aps.append(ap)

    return np.mean(aps) if aps else 0

def calculate_recall(model, dataloader, device, iou_thres=0.5):
    """计算召回率"""
    model.eval()
    total_targets = 0
    total_matched = 0

    with torch.no_grad():
        for imgs, boxes in tqdm(dataloader, desc='Calculating Recall'):
            imgs = imgs.to(device)
            outputs = model(imgs)

            for i in range(len(imgs)):
                pred_boxes = outputs[i].cpu().numpy() if len(outputs) > i else []
                target_boxes = boxes[i].cpu().numpy() if len(boxes) > i else []

                total_targets += len(target_boxes)
                if len(pred_boxes) == 0 or len(target_boxes) == 0:
                    continue

                target_matched = np.zeros(len(target_boxes))
                for pred in pred_boxes:
                    best_iou = 0
                    best_idx = -1
                    for j, target in enumerate(target_boxes):
                        if target_matched[j]:
                            continue
                        iou = calculate_iou(pred[:4], target[:4])
                        if iou > best_iou and iou >= iou_thres:
                            best_iou = iou
                            best_idx = j
                    if best_idx != -1:
                        target_matched[best_idx] = 1

                total_matched += np.sum(target_matched)

    return total_matched / (total_targets + 1e-8) if total_targets > 0 else 0

def get_model_params(model):
    """计算模型参数量（M）"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6

def get_model_flops(model, input_size):
    """计算模型GFLOPs"""
    input_tensor = torch.randn(input_size).to(next(model.parameters()).device)
    flops, _ = profile(model, inputs=(input_tensor,))
    return flops / 1e9
