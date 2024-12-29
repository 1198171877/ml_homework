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
from utils.miou_acc import calculate_metrics

def test(testloader,args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    num_classes = args.num_classes
    # Get attention type
    attention = args.atten

    test_checkpoint = args.test_checkpoint

    # Model setup
    if attention is not None:
        model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    else :
        model = UNetWithAttention(in_channels=3,num_classes=args.num_classes,attention_type=attention).to(device)

    for image,mask in tqdm(testloader):
        with torch.no_grad():

            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1).long()
            outputs = model(images)
            # Compute metrics
            preds = torch.argmax(outputs, dim=1)
            miou, accuracy = calculate_metrics(preds, masks, num_classes)
            

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    test_dataset = SegmentationDataset(args.dataset_path, split='test')
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)






