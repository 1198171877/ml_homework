import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to the dataset root directory.
            split (str): Either 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied
                on a sample (image and mask).
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        
        self.image_paths = sorted(os.listdir(self.image_dir))
        self.mask_paths = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Error loading image or mask: {img_path}, {mask_path}")

        # Convert to RGB (if needed) and normalize to [0, 1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        mask = mask / 255.0  # Assuming masks are binary or single-channel

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dim

        return image, mask

# Example usage
if __name__ == "__main__":
    root_dir = "datasets/SOS/palsar"  # Replace with your dataset root directory
    train_dataset = SegmentationDataset(root_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    for images, masks in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of masks shape: {masks.shape}")
        break
