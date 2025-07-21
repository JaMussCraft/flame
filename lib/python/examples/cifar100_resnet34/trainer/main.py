"""CIFAR-100 ResNet34 horizontal FL trainer for PyTorch.

This example demonstrates distributed training of ResNet34 on CIFAR-100 dataset
using horizontal federated learning with tensor parallelism on BasicBlocks.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
from flame.config import Config
from flame.mode.horizontal.syncfl.trainer import Trainer  
from torchvision import datasets, transforms
import argparse
import random
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HorizontallySplitBasicBlock(nn.Module):
    """BasicBlock with horizontal tensor parallelism.
    
    Split strategy:
    - conv1: split output channels
    - conv2: split input channels, produce full output
    - bn1: split parameters (follows conv1)
    - bn2: full parameters (follows conv2)
    - shortcut: full computation (replicated)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, rank=0, world_size=1):
        super(HorizontallySplitBasicBlock, self).__init__()
        
        self.rank = rank
        self.world_size = world_size
        
        # First conv: split output channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // world_size, 
                              kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // world_size)
        
        # Second conv: split input channels, produce full output
        self.conv2 = nn.Conv2d(out_channels // world_size, out_channels, 
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection - no splitting, each trainer computes full shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Shortcut - full computation (replicated across trainers)
        identity = self.shortcut(x)
        
        # Main path
        out = self.conv1(x)           # Split output
        out = self.bn1(out)           # Split parameters
        out = self.relu(out)
        out = self.conv2(out)         # Split input → full output
        out = self.bn2(out)           # Full parameters
        
        out += identity               # Full + full = full
        out = self.relu(out)
        
        return out


class HorizontallySplitResNet34(nn.Module):
    """ResNet34 with horizontal tensor parallelism on BasicBlocks."""
    
    def __init__(self, rank, world_size, num_classes=100, pretrained=False):
        super(HorizontallySplitResNet34, self).__init__()
        
        self.rank = rank
        self.world_size = world_size
        self.in_channels = 64
        
        # Initial conv layer - split output channels
        self.conv1 = nn.Conv2d(3, 64 // world_size, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 // world_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Final FC layer - full input channels (layer4 output is full)
        self.fc = nn.Linear(512, num_classes)
        
        # No pretrained loading here - will be handled by aggregator

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        
        # First block (may have stride > 1)
        layers.append(HorizontallySplitBasicBlock(self.in_channels, out_channels, 
                                                 stride, self.rank, self.world_size))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(HorizontallySplitBasicBlock(self.in_channels, out_channels,
                                                     1, self.rank, self.world_size))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)           # Split output
        x = self.bn1(x)             # Split parameters
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)     # [batch, 512] - full tensor
        x = self.fc(x)              # [batch, 100]
        
        return F.log_softmax(x, dim=1)


class HorizontalSplitTrainer(Trainer):
    """CIFAR-100 ResNet34 Horizontal Split Trainer."""
    
    def __init__(self, config: Config):
        # Set seed for reproducibility
        seed = getattr(config.hyperparameters, 'seed', 42)
        set_seed(seed)
        
        self.config = config
        self.rank = self.config.hyperparameters.rank
        self.world_size = self.config.hyperparameters.world_size
       
        self.dataset_size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = config.hyperparameters.epochs
        
        self.batch_size = self.config.hyperparameters.batch_size
        self.model = HorizontallySplitResNet34(
            self.rank, self.world_size, pretrained=False
        ).to(self.device)
        
        self.lr = self.config.hyperparameters.learning_rate
        self.train_loader = None

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = HorizontallySplitResNet34(
            self.rank, self.world_size, pretrained=False
        ).to(self.device)

    def load_data(self):
        """Load CIFAR-100 training data with ImageNet preprocessing."""
        # Set generator for reproducible data loading
        g = torch.Generator()
        g.manual_seed(getattr(self.config.hyperparameters, 'seed', 42))
        
        # CIFAR-100 with ImageNet preprocessing (resize to 224x224)
        transform = transforms.Compose([
            transforms.Resize(224),  # Resize to ImageNet size
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(224, padding=28),  # Random crop with padding
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)

        # Simple data partitioning - each trainer gets a subset
        samples_per_trainer = len(dataset) // self.world_size
        start_idx = self.rank * samples_per_trainer
        end_idx = start_idx + samples_per_trainer
        
        indices = torch.arange(start_idx, end_idx)
        subset = data_utils.Subset(dataset, indices)
        self.train_loader = data_utils.DataLoader(
            subset, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(
                getattr(self.config.hyperparameters, 'seed', 42) + worker_id
            )
        )

    def train(self) -> None:
        """Train the model."""
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=0.9, 
            weight_decay=1e-4
        )

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self.train_loader.dataset)

    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % 50 == 0:  # Log every 50 batches
                done = batch_idx * len(data)
                total = len(self.train_loader.dataset)
                percent = 100. * batch_idx / len(self.train_loader)
                logger.info(f"epoch: {epoch} [{done}/{total} ({percent:.0f}%)]"
                            f"\tloss: {loss.item():.6f}")

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement this if testing is needed in trainer
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CIFAR-100 ResNet34 Federated Learning Trainer')
    parser.add_argument("config", nargs="?", default="config.json")
    args = parser.parse_args()
    config = Config(args.config)
    t = HorizontalSplitTrainer(config)
    t.compose()
    t.run()
