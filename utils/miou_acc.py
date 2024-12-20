import numpy as np
import torch

def calculate_iou(pred, target, num_classes):
    """
    Calculate Intersection over Union (IoU) for each class.
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask, shape [H, W].
        target (torch.Tensor): Ground truth mask, shape [H, W].
        num_classes (int): Number of classes.

    Returns:
        list: IoU for each class.
    """
    iou_list = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            iou = float('nan')  # Ignore this class
        else:
            iou = intersection / union
        
        iou_list.append(iou)
    
    return iou_list

def calculate_metrics(pred, target, num_classes):
    """
    Calculate mIoU and pixel accuracy for a batch.
    
    Args:
        pred (torch.Tensor): Predicted segmentation masks, shape [B, H, W].
        target (torch.Tensor): Ground truth masks, shape [B, H, W].
        num_classes (int): Number of classes.
    
    Returns:
        float: mIoU (mean IoU) over the batch.
        float: Pixel accuracy over the batch.
    """
    total_iou = np.zeros(num_classes)
    total_pixels = 0
    correct_pixels = 0
    
    for p, t in zip(pred, target):
        iou_list = calculate_iou(p, t, num_classes)
        total_iou += np.array([iou if not np.isnan(iou) else 0 for iou in iou_list])
        
        correct_pixels += (p == t).sum().item()
        total_pixels += t.numel()
    
    miou = np.nanmean(total_iou / len(pred))  # Mean IoU over the batch
    pixel_accuracy = correct_pixels / total_pixels  # Pixel accuracy over the batch
    
    return miou, pixel_accuracy

# Example usage
if __name__ == "__main__":
    num_classes = 2
    batch_size = 8
    height, width = 256, 256
    
    # Simulated predictions and targets
    pred = torch.randint(0, num_classes, (batch_size, height, width))  # Random predictions
    target = torch.randint(0, num_classes, (batch_size, height, width))  # Random targets
    
    miou, pixel_accuracy = calculate_metrics(pred, target, num_classes)
    print(f"mIoU: {miou:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}")
