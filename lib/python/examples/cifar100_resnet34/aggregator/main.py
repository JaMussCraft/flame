"""CIFAR-100 ResNet34 horizontal FL aggregator for PyTorch.

This example demonstrates distributed training of ResNet34 on CIFAR-100 dataset
using federated learning with horizontal tensor parallelism on BasicBlocks.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from copy import deepcopy
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
import argparse
import random
import numpy as np
import time
import os
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


class BasicBlock(nn.Module):
    """Standard BasicBlock for evaluation."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet34(nn.Module):
    """Standard ResNet34 for evaluation."""
    
    def __init__(self, num_classes=100):
        super(ResNet34, self).__init__()
        
        self.in_channels = 64
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, 1))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)


class Cifar100ResNet34Aggregator(TopAggregator):
    """PyTorch CIFAR-100 ResNet34 Aggregator for horizontal tensor parallelism"""

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
        self.lr = self.config.hyperparameters.learning_rate
        self.rounds = self.config.hyperparameters.rounds
        self.pretrain = self.config.hyperparameters.pretrain
        
        # Initialize experiment results tracking
        self.experiment_results = []
        self.experiment_key = (self.world_size, self.lr, self.rounds, seed)

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

    def initialize(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = ResNet34().to(self.device)
        
        # Load pretrained weights and adapt for CIFAR-100
        if self.pretrain:
            self._load_pretrained_weights()

        # Evaluate baseline before training
        self.evaluate(baseline=True)

    def _load_pretrained_weights(self):
        """Load ImageNet pretrained weights and adapt final layer for CIFAR-100."""
        # Load pretrained ResNet34
        pretrained_model = torchvision.models.resnet34(pretrained=True)
        pretrained_dict = pretrained_model.state_dict()
        
        model_dict = self.model.state_dict()
        
        # Copy all pretrained weights except final FC layer
        adapted_dict = {}
        for name, param in model_dict.items():
            if name in pretrained_dict and 'fc' not in name:
                adapted_dict[name] = pretrained_dict[name]
            elif 'fc' in name:
                # Initialize FC layer randomly for CIFAR-100 (100 classes vs 1000)
                adapted_dict[name] = param  # Keep random initialization
        
        # Load adapted weights
        model_dict.update(adapted_dict)
        self.model.load_state_dict(model_dict)
        
        logger.info("Loaded ImageNet pretrained weights, initialized FC layer for CIFAR-100")

    def load_data(self) -> None:
        # Set generator for reproducible data loading
        g = torch.Generator()
        g.manual_seed(getattr(self.config.hyperparameters, 'seed', 42))
        
        # CIFAR-100 with ImageNet preprocessing for test set
        transform = transforms.Compose([
            transforms.Resize(224),  # Resize to ImageNet size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

        dataset = datasets.CIFAR100('./data',
                                   train=False,
                                   download=True,
                                   transform=transform)

        self.test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
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

        # Write results to a file
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
        
        # Save experiment results if this is the final round
        if self._round == self._rounds:
            self._save_experiment_results()

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

            individual_count = count

        # Concatenate weights from all trainers
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

        # send out global model parameters to trainers
        for end in selected_ends:
            if end not in self.trainer_rank.keys():
                self.trainer_rank[end] = self.trainer_count
                self.trainer_count += 1

            logger.debug(f"sending weights to {end}")
            
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
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )

    def _slice_weights(self, state_dict, rank, world_size):
        """Slice weights for distributed training with tensor parallelism."""
        sliced = {}
        
        for name, full_tensor in state_dict.items():
            if name == "conv1.weight":
                # Split by output channels
                slice_size = 64 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "conv1.bias":
                # Split by output channels (if exists)
                slice_size = 64 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "bn1.weight" or name == "bn1.bias" or name == "bn1.running_mean" or name == "bn1.running_var":
                # Split initial BN parameters
                slice_size = 64 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif '.conv1.weight' in name and 'layer' in name:
                # BasicBlock conv1: split output channels
                layer_channels = self._get_layer_channels(name)
                slice_size = layer_channels // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif '.bn1.weight' in name or '.bn1.bias' in name or '.bn1.running_mean' in name or '.bn1.running_var' in name:
                # BasicBlock bn1: split parameters
                if 'layer' in name:
                    layer_channels = self._get_layer_channels(name)
                    slice_size = layer_channels // world_size
                    sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif '.conv2.weight' in name and 'layer' in name:
                # BasicBlock conv2: split input channels
                layer_channels = self._get_layer_channels(name)
                slice_size = layer_channels // world_size
                sliced[name] = full_tensor[:, rank * slice_size:(rank + 1) * slice_size]
            elif '.bn2' in name or '.shortcut' in name:
                # Full parameters for bn2 and shortcut
                sliced[name] = full_tensor
            elif 'num_batches_tracked' in name:
                # Replicate num_batches_tracked parameters (shared across all trainers)
                sliced[name] = full_tensor
            elif 'fc.weight' in name:
                # Full FC input channels (layer4 output is full)
                sliced[name] = full_tensor
            elif 'fc.bias' in name:
                # Full bias for FC
                sliced[name] = full_tensor
            else:
                # Default: keep full tensor
                sliced[name] = full_tensor
                
        return sliced

    def _concatenate_weights(self, trainers_weights: list) -> dict:
        """Concatenate weights from trainers for ResNet34 with tensor parallelism."""
        concated = {}
        
        # Initial conv1: Split by OUTPUT → Concatenate
        weights = [w['conv1.weight'] for w in trainers_weights]
        concated['conv1.weight'] = torch.cat(weights, 0)
        
        # Initial bn1: Split parameters → Concatenate
        for param in ['bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var']:
            weights = [w[param] for w in trainers_weights]
            concated[param] = torch.cat(weights, 0)
        
        # Handle bn1.num_batches_tracked (scalar parameter → replicate from first trainer)
        if 'bn1.num_batches_tracked' in trainers_weights[0]:
            concated['bn1.num_batches_tracked'] = trainers_weights[0]['bn1.num_batches_tracked']
        
        # Handle all BasicBlock layers
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer_channels = self._get_layer_channels_by_name(layer_name)
            num_blocks = self._get_num_blocks(layer_name)
            
            for block_idx in range(num_blocks):
                block_prefix = f"{layer_name}.{block_idx}"
                
                # conv1: Split by OUTPUT → Concatenate
                conv1_key = f"{block_prefix}.conv1.weight"
                weights = [w[conv1_key] for w in trainers_weights]
                concated[conv1_key] = torch.cat(weights, 0)
                
                # bn1: Split parameters → Concatenate
                for param in ['weight', 'bias', 'running_mean', 'running_var']:
                    bn1_key = f"{block_prefix}.bn1.{param}"
                    weights = [w[bn1_key] for w in trainers_weights]
                    concated[bn1_key] = torch.cat(weights, 0)
                
                # Handle bn1.num_batches_tracked (scalar parameter → replicate from first trainer)
                bn1_num_batches_key = f"{block_prefix}.bn1.num_batches_tracked"
                if bn1_num_batches_key in trainers_weights[0]:
                    concated[bn1_num_batches_key] = trainers_weights[0][bn1_num_batches_key]
                
                # conv2: Split by INPUT → Concatenate along input dimension
                conv2_key = f"{block_prefix}.conv2.weight"
                weights = [w[conv2_key] for w in trainers_weights]
                concated[conv2_key] = torch.cat(weights, 1)
                
                # bn2: Full parameters → Average
                for param in ['weight', 'bias', 'running_mean', 'running_var']:
                    bn2_key = f"{block_prefix}.bn2.{param}"
                    weights = [w[bn2_key] for w in trainers_weights]
                    concated[bn2_key] = torch.mean(torch.stack(weights), dim=0)
                
                # Handle bn2.num_batches_tracked (full parameter → replicate from first trainer)
                bn2_num_batches_key = f"{block_prefix}.bn2.num_batches_tracked"
                if bn2_num_batches_key in trainers_weights[0]:
                    concated[bn2_num_batches_key] = trainers_weights[0][bn2_num_batches_key]
                
                # shortcut (if exists): Full parameters → Average
                shortcut_conv_key = f"{block_prefix}.shortcut.0.weight"
                if shortcut_conv_key in trainers_weights[0]:
                    weights = [w[shortcut_conv_key] for w in trainers_weights]
                    concated[shortcut_conv_key] = torch.mean(torch.stack(weights), dim=0)
                    
                    for param in ['weight', 'bias', 'running_mean', 'running_var']:
                        shortcut_bn_key = f"{block_prefix}.shortcut.1.{param}"
                        weights = [w[shortcut_bn_key] for w in trainers_weights]
                        concated[shortcut_bn_key] = torch.mean(torch.stack(weights), dim=0)
                    
                    # Handle shortcut bn num_batches_tracked (full parameter → replicate from first trainer)
                    shortcut_bn_num_batches_key = f"{block_prefix}.shortcut.1.num_batches_tracked"
                    if shortcut_bn_num_batches_key in trainers_weights[0]:
                        concated[shortcut_bn_num_batches_key] = trainers_weights[0][shortcut_bn_num_batches_key]
        
        # Final FC: Full parameters → Average
        weights = [w['fc.weight'] for w in trainers_weights]
        concated['fc.weight'] = torch.mean(torch.stack(weights), dim=0)
        
        # FC bias: Full → Average
        weights = [w['fc.bias'] for w in trainers_weights]
        concated['fc.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        return concated

    def _get_layer_channels(self, param_name):
        """Get the number of channels for a given layer parameter."""
        if 'layer1' in param_name:
            return 64
        elif 'layer2' in param_name:
            return 128
        elif 'layer3' in param_name:
            return 256
        elif 'layer4' in param_name:
            return 512
        return 64

    def _get_layer_channels_by_name(self, layer_name):
        """Get the number of channels for a given layer name."""
        if layer_name == 'layer1':
            return 64
        elif layer_name == 'layer2':
            return 128
        elif layer_name == 'layer3':
            return 256
        elif layer_name == 'layer4':
            return 512
        return 64

    def _get_num_blocks(self, layer_name):
        """Get the number of blocks in a layer."""
        if layer_name == 'layer1':
            return 3
        elif layer_name == 'layer2':
            return 4
        elif layer_name == 'layer3':
            return 6
        elif layer_name == 'layer4':
            return 3
        return 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-100 ResNet34 Federated Learning Aggregator')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = Cifar100ResNet34Aggregator(config)
    a.compose() 
    a.run()
