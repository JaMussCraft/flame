# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""CIFAR-100 AlexNet horizontal FL trainer for PyTorch.

This example demonstrates distributed training of AlexNet on CIFAR-100 dataset
using horizontal federated learning with tensor parallelism.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from flame.config import Config
from flame.mode.horizontal.syncfl.trainer import Trainer  
from torchvision import datasets, transforms
import argparse
import json
from flame.config import Config
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


class HorizontallySplitAlexNet(nn.Module):
    """AlexNet with horizontal splitting for distributed training."""
    
    def __init__(self, rank, world_size):
        super().__init__()
        
        self.rank = rank
        self.world_size = world_size

        # Alternating split pattern for conv layers (1D tensor parallelism)
        # Conv1: Split by OUTPUT channels
        self.conv1 = nn.Conv2d(3, 96//world_size, kernel_size=5, stride=1, padding=2)
        # Conv2: Split by INPUT channels, produces full output
        self.conv2 = nn.Conv2d(96//world_size, 256, kernel_size=5, stride=1, padding=2)
        # Conv3: Split by OUTPUT channels
        self.conv3 = nn.Conv2d(256, 384//world_size, kernel_size=3, stride=1, padding=1)
        # Conv4: Split by INPUT channels, produces full output
        self.conv4 = nn.Conv2d(384//world_size, 384, kernel_size=3, stride=1, padding=1)
        # Conv5: Split by OUTPUT channels
        self.conv5 = nn.Conv2d(384, 256//world_size, kernel_size=3, stride=1, padding=1)
        # Conv6: Split by INPUT channels, produces full output
        self.conv6 = nn.Conv2d(256//world_size, 256, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FC layers with alternating split pattern (1D tensor parallelism)
        # FC1: Split by OUTPUT dimension
        self.fc1 = nn.Linear(256 * 4 * 4, 4096//world_size)
        # FC2: Split by INPUT dimension
        self.fc2 = nn.Linear(4096//world_size, 4096)
        # FC3: Split by OUTPUT dimension
        self.fc3 = nn.Linear(4096, 4096//world_size)
        # FC4: Split by INPUT dimension, outputs to 100 classes
        self.fc4 = nn.Linear(4096//world_size, 100)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Forward pass through split AlexNet."""
        # First conv block: conv1 -> conv2 -> pool
        x = F.relu(self.conv1(x))  # [batch, 96//world_size, 32, 32]
        x = F.relu(self.conv2(x))  # [batch, 256, 32, 32] - full output
        x = self.pool(x)           # [batch, 256, 16, 16]
        
        # Second conv block: conv3 -> conv4 -> pool
        x = F.relu(self.conv3(x))  # [batch, 384//world_size, 16, 16]
        x = F.relu(self.conv4(x))  # [batch, 384, 16, 16] - full output
        x = self.pool(x)           # [batch, 384, 8, 8]
        
        # Third conv block: conv5 -> conv6 -> pool
        x = F.relu(self.conv5(x))  # [batch, 256//world_size, 8, 8]
        x = F.relu(self.conv6(x))  # [batch, 256, 8, 8] - full output
        x = self.pool(x)           # [batch, 256, 4, 4]
        
        # Flatten and FC layers
        x = torch.flatten(x, 1)    # [batch, 256*4*4] = [batch, 4096]
        x = F.relu(self.fc1(x))    # [batch, 4096//world_size]
        x = self.dropout(x)
        x = F.relu(self.fc2(x))    # [batch, 4096] - full output
        x = self.dropout(x)
        x = F.relu(self.fc3(x))    # [batch, 4096//world_size]
        x = self.dropout(x)
        x = self.fc4(x)            # [batch, 100]
        
        return F.log_softmax(x, dim=1)


class HorizontalSplitTrainer(Trainer):
    """CIFAR-100 AlexNet Horizontal Split Trainer."""
    
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
        self.model = HorizontallySplitAlexNet(self.rank, self.world_size).to(self.device)
        # Use Adam optimizer with lower learning rate for AlexNet
        self.lr = self.config.hyperparameters.learning_rate
        self.train_loader = None

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = HorizontallySplitAlexNet(self.rank, self.world_size).to(self.device)

    def load_data(self):
        """Load CIFAR-100 training data."""
        # For now, we'll use the standard CIFAR-100 dataset
        # In a real federated setup, this would load from pre-partitioned data files
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)

        # Simple data partitioning - each trainer gets a subset
        # In practice, this would be more sophisticated
        samples_per_trainer = len(dataset) // self.world_size
        start_idx = self.rank * samples_per_trainer
        end_idx = start_idx + samples_per_trainer
        
        indices = torch.arange(start_idx, end_idx)
        subset = data_utils.Subset(dataset, indices)
        self.train_loader = data_utils.DataLoader(subset, batch_size=self.batch_size, shuffle=True)

    def train(self) -> None:
        """Train the model."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

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
            
            if batch_idx % 50 == 0:  # Log less frequently for AlexNet
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
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.json")
    args = parser.parse_args()
    config = Config(args.config)
    t = HorizontalSplitTrainer(config)
    t.compose()
    t.run()
