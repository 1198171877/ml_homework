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
    print(attention)
    test_checkpoint = args.test_checkpoint

    # Model setup
    if attention is None:
        model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
        print("using unet!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else :
        model = UNetWithAttention(in_channels=3,num_classes=args.num_classes,attention_type=attention).to(device)
        print("using {}".format(attention))
    model.load_state_dict(torch.load(args.test_checkpoint,map_location=device)['model_state_dict'])
    avg_miou,avg_acc = 0,0
    for images,masks in tqdm(testloader):
        with torch.no_grad():

            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1).long()
            outputs = model(images)
            # Compute metrics
            preds = torch.argmax(outputs, dim=1)
            miou, accuracy = calculate_metrics(preds, masks, num_classes)
            avg_miou+=miou
            avg_acc+=accuracy
    avg_acc/=len(testloader.dataset)
    avg_miou/=len(testloader.dataset)
    print(avg_miou, avg_acc)
            

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    test_dataset = SegmentationDataset(args.dataset_path, split='test')
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test(testloader,args=args)






