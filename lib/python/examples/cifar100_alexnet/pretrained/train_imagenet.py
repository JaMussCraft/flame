"""
Training script for AlexNet on ImageNet subset.
Trains the custom AlexNet architecture from scratch.
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# Add parent directory to path to import AlexNet
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'aggregator'))
from main import AlexNet as CIFAR100AlexNet, set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImageNetAlexNet(nn.Module):
    """AlexNet designed for ImageNet (224x224 input)."""
    
    def __init__(self, num_classes=1000):
        super(ImageNetAlexNet, self).__init__()
        
        # Convolutional layers for 224x224 input
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 224->55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55->27
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 27->27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27->13
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 13->13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 13->13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 13->13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13->6
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class AlexNet(nn.Module):
    """Wrapper class to choose between CIFAR-100 and ImageNet AlexNet."""
    
    def __init__(self, num_classes=100, architecture='cifar100'):
        super(AlexNet, self).__init__()
        self.architecture = architecture
        
        if architecture == 'imagenet':
            self.model = ImageNetAlexNet(num_classes)
        elif architecture == 'cifar100':
            self.model = CIFAR100AlexNet(num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}. Choose 'imagenet' or 'cifar100'")
    
    def forward(self, x):
        return self.model(x)


class ImageNetSubset(datasets.ImageFolder):
    """ImageNet subset dataset class."""
    
    def __init__(self, root: str, selected_classes: list = None, subset_classes: int = 100, **kwargs):
        """
        Initialize ImageNet subset.
        
        Args:
            root: Path to ImageNet dataset
            selected_classes: List of specific class names to use. If None, will select first subset_classes
            subset_classes: Number of classes to use (default 100) - only used if selected_classes is None
            **kwargs: Additional arguments for ImageFolder
        """
        super().__init__(root, **kwargs)
        
        # Select subset of classes
        if selected_classes is not None:
            # Use provided class list
            selected_classes = [cls for cls in selected_classes if cls in self.classes]
            if len(selected_classes) == 0:
                raise ValueError("None of the selected classes found in dataset")
        elif subset_classes < len(self.classes):
            # Select first N classes if no specific list provided
            selected_classes = sorted(self.classes)[:subset_classes]
        else:
            # Use all classes
            selected_classes = self.classes
        
        class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        
        # Filter samples
        samples = []
        for path, target in self.samples:
            class_name = self.classes[target]
            if class_name in selected_classes:
                new_target = class_to_idx[class_name]
                samples.append((path, new_target))
        
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = selected_classes
        self.class_to_idx = class_to_idx


class ImageNetTrainer:
    """ImageNet trainer for custom AlexNet."""
    
    def __init__(self, config: Dict):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed for reproducibility
        set_seed(config['seed'])
        
        # Initialize model
        architecture = config.get('architecture', 'cifar100')
        self.model = AlexNet(num_classes=config['num_classes'], architecture=architecture).to(self.device)
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_val_acc = 0.0
        self.start_epoch = 0
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_data_loaders(self) -> Tuple[data.DataLoader, data.DataLoader]:
        """Setup train and validation data loaders."""
        
        # Get architecture from config
        architecture = self.config.get('architecture', 'cifar100')
        
        # Set transforms based on architecture
        if architecture == 'imagenet':
            # ImageNet normalization values and 224x224 input
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            # Training transforms with augmentation for ImageNet
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                normalize,
            ])
            
            # Validation transforms for ImageNet
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:  # cifar100 architecture
            # CIFAR-100 normalization values and 32x32 input
            normalize = transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
            
            # Training transforms with augmentation - resize to 32x32 for CIFAR-100 AlexNet
            train_transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Force resize to exactly 32x32
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                normalize,
            ])
            
            # Validation transforms - resize to 32x32 for CIFAR-100 model
            val_transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Force resize to exactly 32x32
                transforms.ToTensor(),
                normalize,
            ])
        
        # First, determine common classes between train and val directories
        train_path = os.path.join(self.config['data_path'], 'train')
        val_path = os.path.join(self.config['data_path'], 'val')
        
        # Get available classes in each directory
        train_classes = set([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        val_classes = set([d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))])
        
        # Find common classes
        common_classes = sorted(list(train_classes.intersection(val_classes)))
        
        if len(common_classes) == 0:
            raise ValueError("No common classes found between train and validation sets")
        
        # Select subset of common classes
        num_classes_to_use = min(self.config['num_classes'], len(common_classes))
        selected_classes = common_classes[:num_classes_to_use]
        
        logger.info(f"Found {len(train_classes)} training classes, {len(val_classes)} validation classes")
        logger.info(f"Common classes: {len(common_classes)}")
        logger.info(f"Using {len(selected_classes)} classes for training and validation")
        logger.info(f"Selected classes: {selected_classes[:10]}..." if len(selected_classes) > 10 else f"Selected classes: {selected_classes}")
        
        # Create datasets with the same selected classes
        train_dataset = ImageNetSubset(
            root=train_path,
            selected_classes=selected_classes,
            transform=train_transform
        )
        
        val_dataset = ImageNetSubset(
            root=val_path,
            selected_classes=selected_classes,
            transform=val_transform
        )
        
        # Verify both datasets have the same classes
        if train_dataset.classes != val_dataset.classes:
            raise ValueError("Train and validation datasets have different classes!")
        
        # Create data loaders
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Number of classes: {len(train_dataset.classes)}")
        
        return train_loader, val_loader
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        if self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config['scheduler'] == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=self.config['step_size'],
                gamma=self.config['gamma']
            )
        elif self.config['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Log progress
            if batch_idx % self.config['log_interval'] == 0:
                logger.info(
                    f'Epoch {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}'
                )
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        logger.info(
            f'Epoch {epoch} Training - Loss: {avg_loss:.6f}, '
            f'Accuracy: {accuracy:.2f}% ({correct}/{total}), '
            f'Time: {epoch_time:.2f}s'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'time': epoch_time
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        logger.info(
            f'Epoch {epoch} Validation - Loss: {avg_loss:.6f}, '
            f'Accuracy: {accuracy:.2f}% ({correct}/{total}), '
            f'Time: {epoch_time:.2f}s'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'time': epoch_time
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['output_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f'New best model saved with validation accuracy: {self.best_val_acc:.2f}%')
        
        logger.info(f'Checkpoint saved: {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint['scheduler_state_dict'] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            
            logger.info(f'Checkpoint loaded: {checkpoint_path}')
            logger.info(f'Resuming from epoch {self.start_epoch}')
        else:
            logger.warning(f'Checkpoint not found: {checkpoint_path}')
    
    def train(self):
        """Main training loop."""
        logger.info('Starting training...')
        
        training_history = []
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            logger.info(f'Epoch {epoch}/{self.config["epochs"]-1}')
            logger.info(f'Learning rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
            
            # Save checkpoint every few epochs and if best
            if epoch % self.config['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Record metrics
            epoch_metrics = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            training_history.append(epoch_metrics)
            
            # Save training history
            history_path = os.path.join(self.config['output_dir'], 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
        
        logger.info('Training completed!')
        logger.info(f'Best validation accuracy: {self.best_val_acc:.2f}%')


def validate_imagenet_structure(data_path: str) -> bool:
    """Validate ImageNet dataset structure."""
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    
    if not os.path.exists(train_path):
        logger.error(f"Training directory not found: {train_path}")
        return False
    
    if not os.path.exists(val_path):
        logger.error(f"Validation directory not found: {val_path}")
        return False
    
    # Check if directories contain class folders
    train_classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    val_classes = [d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))]
    
    if len(train_classes) == 0:
        logger.error("No class directories found in training set")
        return False
    
    if len(val_classes) == 0:
        logger.error("No class directories found in validation set")
        return False
    
    logger.info(f"Found {len(train_classes)} training classes and {len(val_classes)} validation classes")
    return True


def get_default_config() -> Dict:
    """Get default training configuration."""
    return {
        # Data
        'data_path': '/path/to/imagenet',  # Update this path
        'num_classes': 100,
        'batch_size': 128,
        'num_workers': 4,
        
        # Model
        'architecture': 'imagenet',  # 'imagenet' or 'cifar100'
        
        # Training
        'epochs': 100,
        'learning_rate': 0.01,
        'optimizer': 'sgd',  # 'sgd', 'adam', 'adamw'
        'momentum': 0.9,
        'weight_decay': 1e-4,
        
        # Scheduler
        'scheduler': 'step',  # 'step', 'cosine', None
        'step_size': 30,
        'gamma': 0.1,
        
        # Logging and saving
        'log_interval': 100,
        'save_interval': 10,
        'output_dir': './checkpoints',
        
        # Reproducibility
        'seed': 42,
    }


def main():
    parser = argparse.ArgumentParser(description='Train AlexNet on ImageNet subset')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--data-path', type=str, help='Path to ImageNet dataset')
    parser.add_argument('--num-classes', type=int, default=100, help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'none'])
    parser.add_argument('--architecture', type=str, default='imagenet', choices=['imagenet', 'cifar100'], 
                        help='Architecture to use: imagenet (224x224) or cifar100 (32x32)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.data_path:
        config['data_path'] = args.data_path
    if args.num_classes:
        config['num_classes'] = args.num_classes
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.optimizer:
        config['optimizer'] = args.optimizer
    if args.scheduler and args.scheduler != 'none':
        config['scheduler'] = args.scheduler
    elif args.scheduler == 'none':
        config['scheduler'] = None
    if args.architecture:
        config['architecture'] = args.architecture
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Validate dataset structure
    if not validate_imagenet_structure(config['data_path']):
        logger.error("Please download and organize ImageNet dataset first!")
        logger.error("Expected structure:")
        logger.error("  /path/to/imagenet/")
        logger.error("  ├── train/")
        logger.error("  │   ├── n01440764/")
        logger.error("  │   │   ├── image1.JPEG")
        logger.error("  │   │   └── ...")
        logger.error("  │   └── ...")
        logger.error("  └── val/")
        logger.error("      ├── n01440764/")
        logger.error("      └── ...")
        return
    
    # Create trainer
    trainer = ImageNetTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
