"""CIFAR-100 AlexNet horizontal FL aggregator for PyTorch.

This example demonstrates distributed training of AlexNet on CIFAR-100 dataset
using federated learning with horizontal tensor parallelism.
"""

import logging
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator
from torchvision import datasets, transforms
from flame.mode.message import MessageType
from flame.common.util import weights_to_model_device
from flame.optimizer.train_result import TrainResult
from flame.common.util import (MLFramework, get_ml_framework_in_use,
                               valid_frameworks, weights_to_device,
                               weights_to_model_device)

from flame.common.constants import DeviceType
from datetime import datetime
import torch.utils.data as data_utils

import os
import numpy as np
from PIL import Image
import random
import pickle


logger = logging.getLogger(__name__)

PROP_ROUND_START_TIME = "round_start_time"


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AlexNet(nn.Module):
    """AlexNet class for CIFAR-100."""

    def __init__(self, num_classes=100):
        """Initialize AlexNet with 6 conv layers and 4 FC layers.

        Using even number of layers to avoid 2D tensor parallelism
        
        
        """
        super(AlexNet, self).__init__()
        
        # 6 Convolutional layers with specified kernel sizes
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)   # 5x5 kernel
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)  # 5x5 kernel
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1) # 3x3 kernel
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1) # 3x3 kernel
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1) # 3x3 kernel
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 3x3 kernel
        
        # Max pooling layers (applied after conv pairs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4 FC layers
        # Input: 32x32, after 3 pooling operations: 32->16->8->4
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)   # 256 channels, 4x4 spatial
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First conv block: conv1 -> conv2 -> pool
        x = F.relu(self.conv1(x))  # Output shape: [batch, 96, 32, 32]
        x = F.relu(self.conv2(x))  # Output shape: [batch, 256, 32, 32]
        x = self.pool(x)           # Output shape: [batch, 256, 16, 16]
        
        # Second conv block: conv3 -> conv4 -> pool
        x = F.relu(self.conv3(x))  # Output shape: [batch, 384, 16, 16]
        x = F.relu(self.conv4(x))  # Output shape: [batch, 384, 16, 16]
        x = self.pool(x)           # Output shape: [batch, 384, 8, 8]
        
        # Third conv block: conv5 -> conv6 -> pool
        x = F.relu(self.conv5(x))  # Output shape: [batch, 256, 8, 8]
        x = F.relu(self.conv6(x))  # Output shape: [batch, 256, 8, 8]
        x = self.pool(x)           # Output shape: [batch, 256, 4, 4]
        
        # Flatten and FC layers
        x = torch.flatten(x, 1)    # Output shape: [batch, 256*4*4] = [batch, 4096]
        x = F.relu(self.fc1(x))    # Output shape: [batch, 4096]
        x = self.dropout(x)
        x = F.relu(self.fc2(x))    # Output shape: [batch, 4096]
        x = self.dropout(x)
        x = F.relu(self.fc3(x))    # Output shape: [batch, 4096]
        x = self.dropout(x)
        x = self.fc4(x)            # Output shape: [batch, 100]
        
        return F.log_softmax(x, dim=1)


class Cifar100AlexNetAggregator(TopAggregator):
    """PyTorch CIFAR-100 AlexNet Aggregator for horizontal tensor parallelism"""

    def __init__(self, config: Config) -> None:
        # Set seed for reproducibility
        seed = getattr(config.hyperparameters, 'seed', 42)
        set_seed(seed)
        
        self.config = config
        self.model = None
        self.dataset: Dataset = None
        self.device = None
        self.test_loader = None
        self.trainer_rank = {}
        self.trainer_count = 0
        self.world_size = self.config.hyperparameters.world_size
        self.all_available = False
        self.con_time = 0
        self.enable_swapping = self.config.hyperparameters.enable_swapping
        self.enable_layerwise_swapping = self.config.hyperparameters.enable_layerwise_swapping 
        self.lr = self.config.hyperparameters.learning_rate
        self.rounds = self.config.hyperparameters.rounds
        
        # Initialize experiment results tracking
        self.experiment_results = []
        self.experiment_key = (self.world_size, self.lr, self.enable_swapping, self.enable_layerwise_swapping, self.rounds, seed)
    
    def _save_experiment_results(self):
        """Save experiment results to a pickle file."""
        if not self.experiment_results:
            return
            
        results_file = "experiment_results.pkl"
        results_dict = {}
        
        # Load existing results if file exists
        if os.path.exists(results_file):
            try:
                with open(results_file, 'rb') as f:
                    results_dict = pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
                results_dict = {}
        
        # Add current experiment results
        results_dict[self.experiment_key] = self.experiment_results
        
        # Save updated results
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(results_dict, f)
            logger.info(f"Saved experiment results to {results_file}")
        except Exception as e:
            logger.error(f"Could not save experiment results: {e}")

    def _save_model_checkpoint(self, test_loss, test_accuracy):
        """Save model checkpoint with metadata."""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_weights': self.weights,
                'round': self._round,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'world_size': self.world_size,
                'learning_rate': self.lr,
                'enable_swapping': self.enable_swapping,
                'experiment_key': self.experiment_key,
                'concatenation_time': self.con_time
            }
            
            # Create checkpoint filename with experiment parameters
            checkpoint_filename = f"model_checkpoint_ws{self.world_size}_lr{self.lr}_swap{self.enable_swapping}_r{self._rounds}.pth"
            
            torch.save(checkpoint, checkpoint_filename)
            logger.info(f"Saved model checkpoint to {checkpoint_filename}")
            
        except Exception as e:
            logger.error(f"Could not save model checkpoint: {e}")

    def initialize(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = AlexNet().to(self.device)

        # Load pretrained weights (pretrained on CIFAR100)
        self.model.load_state_dict(torch.load('/home/cc/cc_my_mounting_point/james-imagenet/checkpoint_pretrain_cifar100_lr0.0001_r10.pth')["model_state_dict"])   
        logger.info("Loaded pretrained weights!")

        # Evaluate baseline before
        self.evaluate(baseline=True)

        # TODO: start with a pretrained model (maybe pretrain using the same dataset or ImageNet)

    def load_data(self) -> None:
        # Set generator for reproducible data loading
        g = torch.Generator()
        g.manual_seed(getattr(self.config.hyperparameters, 'seed', 42))
        
        # CIFAR-100 specific normalization values
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        dataset = datasets.CIFAR100('./data',
                                   train=False,
                                   download=True,
                                   transform=transform)

        self.test_loader = torch.utils.data.DataLoader(
            dataset,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(
                getattr(self.config.hyperparameters, 'seed', 42) + worker_id
            )
        )

        # store data into dataset for analysis (e.g., bias)
        self.dataset = Dataset(dataloader=self.test_loader)

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass

    def evaluate(self, baseline=False) -> None:
        """Evaluate (test) a model."""
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(
                    output, target,
                    reduction='sum').item()  # sum up batch loss
                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        total = len(self.test_loader.dataset)
        test_loss /= total
        test_accuracy = correct / total


        # Store result for pickle saving
        self.experiment_results.append((self._round if not baseline else 0, test_loss, test_accuracy))

        # Write results to a file instead of logging to terminal (keep for compatibility)
        with open(f'eval_res_ws{self.world_size}_r{self._rounds}.txt', 'a') as f:
            f.write(f"Round: {self._round if not baseline else 0}\n")
            f.write(f"Concatenation Time: {self.con_time}\n")
            f.write(f"Test loss: {test_loss}\n")
            f.write(f"Test accuracy: {correct}/{total} ({test_accuracy})\n")
            f.write("\n")  # Add a blank line for readability
        
        logger.info(f"Evaluated round {self._round if not baseline else 0}!")
        
        # Update metrics for model registry
        self.update_metrics({
            'test-loss': test_loss,
            'test-accuracy': test_accuracy
        })
        
        # Save experiment results and model checkpoint if this is the final round
        if self._round == self._rounds:
            self._save_experiment_results()
            # self._save_model_checkpoint(test_loss, test_accuracy)

    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        trainers_weights = [0] * self.world_size
        count_total = 0
        individual_count = 0
        w_received = 0

        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.info(f"Received data from {end}")

            weights = None
            count = 0

            if MessageType.WEIGHTS in msg:
                weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            if MessageType.DATASAMPLER_METADATA in msg:
                self.datasampler.handle_metadata_from_trainer(
                    msg[MessageType.DATASAMPLER_METADATA], end, channel
                )

            logger.debug(f"{end}'s parameters trained with {count} samples")

            if weights is not None and count > 0:
                count_total += count
                w_received += 1
                trainers_weights[self.trainer_rank[end]] = weights

            individual_count = count        # Concatenate weights from all trainers
        if w_received == self.world_size:
            start_time = time.time()
            
            concated = self._concatenate_weights(trainers_weights)

            tres = TrainResult(concated, count_total)
            self.cache["concat"] = tres
            end_time = time.time()
            self.con_time = end_time - start_time

        logger.debug(f"Received and collected weights from {len(channel.ends())} trainers")
        
        if count_total == individual_count * self.world_size:
            logger.info('Weight aggregation successful')
            global_weights = self.optimizer.do(
                deepcopy(self.weights),
                self.cache,
                total=count_total,
                num_trainers=len(channel.ends()),
            )
            if global_weights is None:
                logger.info("Failed model aggregation")
                time.sleep(1)
                return
            self.weights = global_weights
            self._update_model()

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        while 1:
            list_ends = channel.all_ends()
            n_ends = len(list_ends)
            time.sleep(3)

            logger.info(f"# of joined ends: {n_ends}")

            if n_ends == self.world_size:
                logger.info(f"Begin weights distribution with {n_ends} joined ends")
                break

        # before distributing weights, update it from global model
        self._update_weights()

        selected_ends = channel.ends()
        datasampler_metadata = self.datasampler.get_metadata(self._round, selected_ends)

        # Swapping logic (only if enabled)
        if self.enable_swapping and self._round % 2 == 0 and self.trainer_count == self.world_size:
            for key in self.trainer_rank:
                self.trainer_rank[key] = (self.trainer_rank[key] + 1) % self.world_size

        # send out global model parameters to trainers
        for end in selected_ends:
            if end not in self.trainer_rank.keys():
                self.trainer_rank[end] = self.trainer_count
                self.trainer_count += 1

            logger.debug(f"sending weights to {end}")
            
            # Apply layerwise swapping for specific layers if enabled
            if self.enable_layerwise_swapping:
                effective_rank = self._get_effective_rank(self.trainer_rank[end])
                temp = self._slice_weights(self.weights, self.trainer_rank[end], self.world_size, effective_rank)
            else:
                temp = self._slice_weights(self.weights, self.trainer_rank[end], self.world_size)
            
            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        temp, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                    MessageType.DATASAMPLER_METADATA: datasampler_metadata,
                },
            )
            # register round start time on each end for round duration measurement.
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (round, datetime.now())
            )

    def _slice_weights(self, state_dict, rank, world_size, effective_rank=None):
        """Slice weights for distributed training with 1D tensor parallelism and optional layerwise swapping."""
        if effective_rank is None:
            effective_rank = {}
        
        sliced = {}
        for name, full_tensor in state_dict.items():
            if name == "conv1.weight":
                # Split by output channels
                slice_size = 96 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "conv1.bias":
                # Split by output channels
                slice_size = 96 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "conv2.weight":
                # Split by input channels - use effective rank for layerwise swapping if enabled
                effective_r = effective_rank.get('conv2', rank)
                in_slice_size = 96 // world_size
                sliced[name] = full_tensor[:, effective_r * in_slice_size:(effective_r + 1) * in_slice_size]
            elif name == "conv2.bias":
                # Same for all trainers (input channel split)
                sliced[name] = full_tensor
            elif name == "conv3.weight":
                # Split by output channels
                slice_size = 384 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "conv3.bias":
                # Split by output channels
                slice_size = 384 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "conv4.weight":
                # Split by input channels - use effective rank for layerwise swapping if enabled
                effective_r = effective_rank.get('conv4', rank)
                in_slice_size = 384 // world_size
                sliced[name] = full_tensor[:, effective_r * in_slice_size:(effective_r + 1) * in_slice_size]
            elif name == "conv4.bias":
                # Same for all trainers (input channel split)
                sliced[name] = full_tensor
            elif name == "conv5.weight":
                # Split by output channels
                slice_size = 256 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "conv5.bias":
                # Split by output channels
                slice_size = 256 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "conv6.weight":
                # Split by input channels - use effective rank for layerwise swapping if enabled
                effective_r = effective_rank.get('conv6', rank)
                in_slice_size = 256 // world_size
                sliced[name] = full_tensor[:, effective_r * in_slice_size:(effective_r + 1) * in_slice_size]
            elif name == "conv6.bias":
                # Same for all trainers (input channel split)
                sliced[name] = full_tensor
            elif name == "fc1.weight":
                # Split by output dimension
                out_slice_size = 4096 // world_size
                sliced[name] = full_tensor[rank * out_slice_size:(rank + 1) * out_slice_size]
            elif name == "fc1.bias":
                # Split by output dimension
                out_slice_size = 4096 // world_size
                sliced[name] = full_tensor[rank * out_slice_size:(rank + 1) * out_slice_size]
            elif name == "fc2.weight":
                # Split by input dimension - use effective rank for layerwise swapping if enabled
                effective_r = effective_rank.get('fc2', rank)
                in_slice_size = 4096 // world_size
                sliced[name] = full_tensor[:, effective_r * in_slice_size:(effective_r + 1) * in_slice_size]
            elif name == "fc2.bias":
                # Same for all trainers (input dimension split)
                sliced[name] = full_tensor
            elif name == "fc3.weight":
                # Split by output dimension
                out_slice_size = 4096 // world_size
                sliced[name] = full_tensor[rank * out_slice_size:(rank + 1) * out_slice_size]
            elif name == "fc3.bias":
                # Split by output dimension
                out_slice_size = 4096 // world_size
                sliced[name] = full_tensor[rank * out_slice_size:(rank + 1) * out_slice_size]
            elif name == "fc4.weight":
                # Split by input dimension - use effective rank for layerwise swapping if enabled
                effective_r = effective_rank.get('fc4', rank)
                in_slice_size = 4096 // world_size
                sliced[name] = full_tensor[:, effective_r * in_slice_size:(effective_r + 1) * in_slice_size]
            elif name == "fc4.bias":
                # Same for all trainers (input dimension split) - final 100 classes
                sliced[name] = full_tensor
        return sliced

    def _concatenate_weights(self, trainers_weights: list) -> dict:
        """Concatenate weights from trainers for AlexNet with 1D tensor parallelism."""
        concated = {}
        
        # Conv1: Split by OUTPUT → Concatenate weights and bias
        weights = [w['conv1.weight'] for w in trainers_weights]
        concated['conv1.weight'] = torch.cat(weights, 0)
        weights = [w['conv1.bias'] for w in trainers_weights]
        concated['conv1.bias'] = torch.cat(weights, 0)
        
        # Conv2: Split by INPUT → Concatenate weights, Average bias
        weights = [w['conv2.weight'] for w in trainers_weights]
        concated['conv2.weight'] = torch.cat(weights, 1)
        weights = [w['conv2.bias'] for w in trainers_weights]
        concated['conv2.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        # Conv3: Split by OUTPUT → Concatenate weights and bias
        weights = [w['conv3.weight'] for w in trainers_weights]
        concated['conv3.weight'] = torch.cat(weights, 0)
        weights = [w['conv3.bias'] for w in trainers_weights]
        concated['conv3.bias'] = torch.cat(weights, 0)
        
        # Conv4: Split by INPUT → Concatenate weights, Average bias
        weights = [w['conv4.weight'] for w in trainers_weights]
        concated['conv4.weight'] = torch.cat(weights, 1)
        weights = [w['conv4.bias'] for w in trainers_weights]
        concated['conv4.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        # Conv5: Split by OUTPUT → Concatenate weights and bias
        weights = [w['conv5.weight'] for w in trainers_weights]
        concated['conv5.weight'] = torch.cat(weights, 0)
        weights = [w['conv5.bias'] for w in trainers_weights]
        concated['conv5.bias'] = torch.cat(weights, 0)
        
        # Conv6: Split by INPUT → Concatenate weights, Average bias
        weights = [w['conv6.weight'] for w in trainers_weights]
        concated['conv6.weight'] = torch.cat(weights, 1)
        weights = [w['conv6.bias'] for w in trainers_weights]
        concated['conv6.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        # FC1: Split by OUTPUT → Concatenate weights and bias
        weights = [w['fc1.weight'] for w in trainers_weights]
        concated['fc1.weight'] = torch.cat(weights, 0)
        weights = [w['fc1.bias'] for w in trainers_weights]
        concated['fc1.bias'] = torch.cat(weights, 0)
        
        # FC2: Split by INPUT → Concatenate weights, Average bias
        weights = [w['fc2.weight'] for w in trainers_weights]
        concated['fc2.weight'] = torch.cat(weights, 1)
        weights = [w['fc2.bias'] for w in trainers_weights]
        concated['fc2.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        # FC3: Split by OUTPUT → Concatenate weights and bias
        weights = [w['fc3.weight'] for w in trainers_weights]
        concated['fc3.weight'] = torch.cat(weights, 0)
        weights = [w['fc3.bias'] for w in trainers_weights]
        concated['fc3.bias'] = torch.cat(weights, 0)
        
        # FC4: Split by INPUT → Concatenate weights, Average bias
        weights = [w['fc4.weight'] for w in trainers_weights]
        concated['fc4.weight'] = torch.cat(weights, 1)
        weights = [w['fc4.bias'] for w in trainers_weights]
        concated['fc4.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        return concated

    def _get_effective_rank(self, trainer_rank):
        """Get effective rank for layerwise swapping of specific layers."""
        if not self.enable_layerwise_swapping:
            return {}
        
        # Layers to apply layerwise swapping: conv2, conv4, conv6, fc2, fc4
        swappable_layers = ['conv2', 'conv4', 'conv6', 'fc2', 'fc4']
        
        effective_rank = {}
        for layer_name in swappable_layers:
            # Apply swapping every round (not just even rounds)
            if self._round > 0:
                # Shift rank by 1 for swappable layers
                effective_rank[layer_name] = (trainer_rank + self._round) % self.world_size
            else:
                effective_rank[layer_name] = trainer_rank
        
        return effective_rank


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CIFAR-100 AlexNet Federated Learning Aggregator')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = Cifar100AlexNetAggregator(config)
    a.compose() 
    a.run()
