import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from utils.parser import get_parser
from tqdm import tqdm
from torch.utils import tensorboard
from models.Dataloader import SegmentationDataset
from models.Unet import UNet

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    writer.add_scalar('Train Loss', avg_loss, epoch)
    return avg_loss

def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            validation_loss += loss.item()

    avg_loss = validation_loss / len(dataloader)
    writer.add_scalar('Validation Loss', avg_loss, epoch)
    return avg_loss

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TensorBoard writer
    writer = tensorboard.SummaryWriter()

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load datasets
    train_dataset = SegmentationDataset(args.dataset_path, split='train', transform=transform)
    test_dataset = SegmentationDataset(args.dataset_path, split='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Model setup
    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)

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
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss = validate(model, test_loader, criterion, device, epoch, writer)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.save_model_path)

    writer.close()

if __name__ == "__main__":
    main()
