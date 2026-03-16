"""
YOLO 目标检测评估工具
计算 mAP@0.5、Precision、Recall
用于对比模型预测结果和人工标注的真实标签
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


def read_yolo_labels(label_path: str) -> List[List[float]]:
    """
    读取 YOLO 格式的标注文件
    
    Args:
        label_path: 标注文件路径
        
    Returns:
        boxes: [[class_id, x_center, y_center, width, height, confidence], ...]
               如果没有 confidence，默认为 1.0
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    # 标准 YOLO 格式，没有置信度，默认为 1.0
                    boxes.append([float(x) for x in parts] + [1.0])
                elif len(parts) == 6:
                    # 带置信度的格式
                    boxes.append([float(x) for x in parts])
    return boxes


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个边界框的 IoU (Intersection over Union)
    
    Args:
        box1, box2: [class_id, x_center, y_center, width, height, confidence] (归一化坐标)
        
    Returns:
        iou: IoU 值
    """
    # 提取坐标 (忽略 class_id 和 confidence)
    x1, y1, w1, h1 = box1[1:5]
    x2, y2, w2, h2 = box2[1:5]
    
    # 转换为左上角和右下角坐标
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # 交集面积
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # 并集面积
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def match_boxes(pred_boxes: List[List[float]], 
                gt_boxes: List[List[float]], 
                iou_threshold: float = 0.5) -> Tuple[int, int, int]:
    """
    匹配预测框和真实框，计算 TP, FP, FN
    
    Args:
        pred_boxes: 预测框列表
        gt_boxes: 真实框列表
        iou_threshold: IoU 阈值
        
    Returns:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
    """
    tp = 0
    fp = 0
    fn = 0
    
    matched_gt = set()  # 记录已匹配的真实框
    
    # 遍历每个预测框
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        # 找到与预测框 IoU 最大的真实框
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # 判断是否匹配成功
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    # 未匹配的真实框为 FN
    fn = len(gt_boxes) - len(matched_gt)
    
    return tp, fp, fn


def calculate_ap_voc(pred_boxes_all: List[List[float]], 
                     gt_boxes_all: List[List[float]],
                     gt_counts_per_image: List[int],
                     iou_threshold: float = 0.5) -> Tuple[float, List[float], List[float]]:
    """
    使用 PASCAL VOC 方法计算 Average Precision
    
    Args:
        pred_boxes_all: 所有图片的预测框 [[class_id, x, y, w, h, confidence, img_id], ...]
        gt_boxes_all: 所有图片的真实框 [[class_id, x, y, w, h, confidence, img_id], ...]
        gt_counts_per_image: 每张图片的真实框数量
        iou_threshold: IoU 阈值
        
    Returns:
        ap: Average Precision
        precisions: Precision 列表
        recalls: Recall 列表
    """
    # 按置信度降序排序所有预测框
    pred_boxes_sorted = sorted(pred_boxes_all, key=lambda x: x[5], reverse=True)
    
    total_gt = sum(gt_counts_per_image)
    if total_gt == 0:
        return 0.0, [], []
    
    # 为每张图片的每个 GT 框创建匹配标记
    gt_matched = {}
    for box in gt_boxes_all:
        img_id = int(box[6])
        if img_id not in gt_matched:
            gt_matched[img_id] = set()
    
    tp_list = []
    fp_list = []
    
    # 遍历每个预测框
    for pred_box in pred_boxes_sorted:
        pred_img_id = int(pred_box[6])
        
        # 找到同一张图片中与预测框 IoU 最大的真实框
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes_all):
            gt_img_id = int(gt_box[6])
            
            # 只匹配同一张图片的框
            if gt_img_id != pred_img_id:
                continue
            
            # 跳过已匹配的 GT 框
            if gt_idx in gt_matched.get(gt_img_id, set()):
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # 判断是否匹配成功
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[pred_img_id].add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)
    
    # 计算累积 TP 和 FP
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    
    # 计算 Precision 和 Recall
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gt
    
    # 使用 11-point interpolation 计算 AP (PASCAL VOC 方法)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        # 找到 recall >= t 的最大 precision
        p = 0.0
        for i in range(len(recalls)):
            if recalls[i] >= t:
                p = max(p, precisions[i])
        ap += p / 11.0
    
    return ap, precisions.tolist(), recalls.tolist()


def calculate_metrics_correct(pred_dir: str, 
                              gt_dir: str,
                              iou_threshold: float = 0.5) -> Dict:
    """
    正确计算评估指标（跨图片的 mAP）
    
    Args:
        pred_dir: 预测标签目录
        gt_dir: 真实标签目录
        iou_threshold: IoU 阈值
        
    Returns:
        metrics: 包含各项指标的字典
    """
    pred_files = list(Path(pred_dir).glob('*.txt'))
    print(f"找到 {len(pred_files)} 个预测文件")
    
    all_pred_boxes = []
    all_gt_boxes = []
    gt_counts_per_image = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    file_stats = []
    
    for img_id, pred_file in enumerate(pred_files):
        # 读取预测框
        pred_boxes = read_yolo_labels(str(pred_file))
        
        # 读取对应的真实框
        gt_file = Path(gt_dir) / pred_file.name
        gt_boxes = read_yolo_labels(str(gt_file))
        
        # 为每个框添加图片 ID
        for box in pred_boxes:
            all_pred_boxes.append(box + [img_id])
        
        for box in gt_boxes:
            all_gt_boxes.append(box + [img_id])
        
        gt_counts_per_image.append(len(gt_boxes))
        
        # 简单统计（用于显示）
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        file_stats.append({
            'file': pred_file.name,
            'pred_count': len(pred_boxes),
            'gt_count': len(gt_boxes),
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    # 计算整体 Precision 和 Recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算 mAP@0.5（使用 PASCAL VOC 方法）
    print("\n开始计算 mAP (PASCAL VOC 方法)...")
    ap, precisions, recalls = calculate_ap_voc(
        all_pred_boxes, 
        all_gt_boxes, 
        gt_counts_per_image,
        iou_threshold
    )
    
    print(f"总预测框数: {len(all_pred_boxes)}")
    print(f"总真实框数: {sum(gt_counts_per_image)}")
    if all_pred_boxes:
        confidences = [box[5] for box in all_pred_boxes]
        print(f"置信度范围: {min(confidences):.4f} - {max(confidences):.4f}")
        print(f"置信度唯一值数量: {len(set(confidences))}")
    
    metrics = {
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'mAP@0.5': ap,
        'IoU_threshold': iou_threshold,
        'file_stats': file_stats,
        'pr_curve': {'precisions': precisions, 'recalls': recalls}
    }
    
    return metrics


def print_metrics(metrics: Dict):
    """打印评估指标"""
    print("\n" + "="*50)
    print("目标检测评估结果 (PASCAL VOC 方法)")
    print("="*50)
    print(f"IoU 阈值: {metrics['IoU_threshold']}")
    print(f"True Positives (TP):  {metrics['TP']}")
    print(f"False Positives (FP): {metrics['FP']}")
    print(f"False Negatives (FN): {metrics['FN']}")
    print("-"*50)
    print(f"Precision (精准率):   {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
    print(f"Recall (召回率):      {metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)")
    print(f"F1-Score:            {metrics['F1-Score']:.4f}")
    print(f"mAP@0.5:             {metrics['mAP@0.5']:.4f} ({metrics['mAP@0.5']*100:.2f}%)")
    print("="*50)
    
    # 分析结果
    print("\n结果分析:")
    if metrics['FP'] == 0:
        print("✓ Precision = 100%: 模型没有误检，预测的都是正确的")
    if metrics['FN'] > 0:
        print(f"⚠ Recall = {metrics['Recall']*100:.1f}%: 模型漏检了 {metrics['FN']} 个目标")
    if metrics['Precision'] == 1.0 and metrics['Recall'] < 1.0:
        print("→ 模型比较保守，宁可漏检也不误检")
    
    print(f"\n✓ 使用 PASCAL VOC 11-point interpolation 计算的 mAP@0.5: {metrics['mAP@0.5']:.4f}")
    
    # 显示详细文件统计
    print(f"\n详细文件统计 (共 {len(metrics['file_stats'])} 个文件):")
    print("文件名\t\t预测数\t真实数\tTP\tFP\tFN")
    print("-" * 60)
    for stat in metrics['file_stats'][:10]:  # 只显示前10个
        print(f"{stat['file'][:15]:<15}\t{stat['pred_count']}\t{stat['gt_count']}\t{stat['tp']}\t{stat['fp']}\t{stat['fn']}")
    
    if len(metrics['file_stats']) > 10:
        print(f"... 还有 {len(metrics['file_stats']) - 10} 个文件")
    
    print("="*50 + "\n")


def visualize_metrics(metrics: Dict, save_path: str = None):
    """可视化评估指标"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 左图：TP, FP, FN 柱状图
    ax1 = axes[0]
    categories = ['TP', 'FP', 'FN']
    values = [metrics['TP'], metrics['FP'], metrics['FN']]
    colors = ['green', 'red', 'orange']
    
    ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_ylabel('数量')
    ax1.set_title('检测结果统计')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(values):
        ax1.text(i, v, str(v), ha='center', va='bottom')
    
    # 中图：Precision, Recall, F1, mAP 柱状图
    ax2 = axes[1]
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'mAP@0.5']
    metrics_values = [
        metrics['Precision'],
        metrics['Recall'],
        metrics['F1-Score'],
        metrics['mAP@0.5']
    ]
    
    bars = ax2.bar(metrics_names, metrics_values, color='steelblue', alpha=0.7)
    ax2.set_ylabel('分数')
    ax2.set_title('评估指标 (PASCAL VOC)')
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(metrics_values):
        ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # 右图：Precision-Recall 曲线
    ax3 = axes[2]
    if metrics['pr_curve']['precisions'] and metrics['pr_curve']['recalls']:
        precisions = metrics['pr_curve']['precisions']
        recalls = metrics['pr_curve']['recalls']
        ax3.plot(recalls, precisions, 'b-', linewidth=2, label=f"AP={metrics['mAP@0.5']:.3f}")
        ax3.fill_between(recalls, precisions, alpha=0.2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall 曲线')
        ax3.set_xlim([0, 1.0])
        ax3.set_ylim([0, 1.05])
        ax3.grid(alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No PR curve data', ha='center', va='center')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall 曲线')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # 配置路径
    PRED_DIR = '../yolodata/yolomAP/model_yolo'  # 模型预测结果目录
    GT_DIR = '../yolodata/yolomAP/true'   # 人工标注真实标签目录
    IOU_THRESHOLD = 0.5
    
    # 检查目录是否存在
    if not os.path.exists(PRED_DIR):
        print(f"错误: 预测目录不存在: {PRED_DIR}")
        print("请创建目录并放入模型预测的 YOLO txt 文件")
        print("格式: class_id x_center y_center width height confidence")
        exit(1)
    
    if not os.path.exists(GT_DIR):
        print(f"错误: 真实标签目录不存在: {GT_DIR}")
        print("请创建目录并放入人工标注的 YOLO txt 文件")
        print("格式: class_id x_center y_center width height")
        exit(1)
    
    # 计算评估指标
    print("开始计算评估指标 (PASCAL VOC 方法)...")
    metrics = calculate_metrics_correct(PRED_DIR, GT_DIR, IOU_THRESHOLD)
    
    # 打印结果
    print_metrics(metrics)
    
    # 可视化结果
    visualize_metrics(metrics, save_path='plots/evaluation_metrics.png')

