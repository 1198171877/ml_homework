import argparse

def get_parser():
    """
    Set up argument parser for training hyperparameters.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Training Hyperparameter Configuration")

    # Input image parameters
    parser.add_argument('--image_height', type=int, default=572, help='Input image height')
    parser.add_argument('--image_width', type=int, default=572, help='Input image width')

    # Batch sizes
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for testing')

    # Number of classes
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')

    # Dataset and model paths
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset root directory')
    parser.add_argument('--save_model_path', type=str, default='./model.pth', help='Path to save the trained model')

    # Pretrained weights and resume training
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--resume_training', action='store_true', help='Resume training from a checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint file if resuming training')

    return parser