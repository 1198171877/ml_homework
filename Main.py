import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from utils.parser import get_parser
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from models.Dataloader import SegmentationDataset
from models.Unet import UNet
from models.attention_Unet import UNetWithAttention
import numpy as np

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

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, num_classes):
    model.train()
    running_loss = 0.0
    total_miou = 0.0
    total_accuracy = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        
        masks = masks.squeeze(1).long()
        optimizer.zero_grad()
        outputs = model(images)  
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute metrics
        preds = torch.argmax(outputs, dim=1)
        miou, accuracy = calculate_metrics(preds, masks, num_classes)
        total_miou += miou
        total_accuracy += accuracy

        progress_bar.set_postfix(loss=loss.item(), mIoU=miou, accuracy=accuracy)

    avg_loss = running_loss / len(dataloader.dataset)
    avg_miou = total_miou / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    writer.add_scalar('Train Loss', avg_loss, epoch)
    writer.add_scalar('Train mIoU', avg_miou, epoch)
    writer.add_scalar('Train Accuracy', avg_accuracy, epoch)
    return avg_loss, avg_miou, avg_accuracy

def validate(model, dataloader, criterion, device, epoch, writer, num_classes):
    model.eval()
    validation_loss = 0.0
    total_miou = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1).long()
            outputs = model(images)
            loss = criterion(outputs, masks)
            validation_loss += loss.item()

            # Compute metrics
            preds = torch.argmax(outputs, dim=1)
            miou, accuracy = calculate_metrics(preds, masks, num_classes)
            total_miou += miou
            total_accuracy += accuracy

    avg_loss = validation_loss / len(dataloader.dataset)
    avg_miou = total_miou / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    writer.add_scalar('Validation Loss', avg_loss, epoch)
    writer.add_scalar('Validation mIoU', avg_miou, epoch)
    writer.add_scalar('Validation Accuracy', avg_accuracy, epoch)
    return avg_loss, avg_miou, avg_accuracy

def main():
    parser = get_parser()
    args = parser.parse_args()
    # Get attention type
    attention = args.atten

    # Check checkpoints path existence
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TensorBoard writer
    writer = SummaryWriter()

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load datasets
    train_dataset = SegmentationDataset(args.dataset_path, split='train')#, transform=transform)
    test_dataset = SegmentationDataset(args.dataset_path, split='test')#, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Model setup
    if attention is not None:
        model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    else :
        model = UNetWithAttention(in_channels=3,num_classes=args.num_classes,attention_type=attention).to(device)

    if args.use_pretrained:
        print("Using pretrained weights")
        # Load pretrained weights (modify as needed for your application)
        pass

    if args.resume_training and os.path.isfile(args.checkpoint_path):
        print(f"Resuming training from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state = checkpoint['optimizer_state_dict']
    else:
        optimizer_state = None

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_miou, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, args.num_classes)
        val_loss, val_miou, val_accuracy = validate(model, test_loader, criterion, device, epoch, writer, args.num_classes)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.4f}, Validation mIoU: {val_miou:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if (epoch+1) % args.save_period == 0:
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_model_path, f'epoch_{epoch+1}_checkpoint.pth'))

    writer.close()

if __name__ == "__main__":
    main()
