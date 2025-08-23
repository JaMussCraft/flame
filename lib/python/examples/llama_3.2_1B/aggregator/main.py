"""Llama 3.2 1B horizontal FL aggregator for PyTorch.

This example demonstrates distributed training of Llama 3.2 1B model
using federated learning with horizontal tensor parallelism.
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator
from flame.mode.message import MessageType
from flame.common.util import weights_to_model_device
from flame.optimizer.train_result import TrainResult
from flame.common.util import (MLFramework, get_ml_framework_in_use,
                               valid_frameworks, weights_to_device,
                               weights_to_model_device)
from flame.common.constants import DeviceType

from model_no_fairscale import Transformer
from args import ModelArgs
from tokenizer import Tokenizer

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


class TextDataset(TorchDataset):
    """Simple text dataset for fine-tuning."""
    
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        
        # Load data from file
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.texts.append(data['text'])
        else:
            # Plain text file
            with open(data_path, 'r') as f:
                self.texts = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize and pad/truncate
        tokens = self.tokenizer.encode(text, bos=True, eos=True)
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Pad with tokenizer pad token if available, otherwise use eos
            pad_token = getattr(self.tokenizer, 'pad_id', self.tokenizer.eos_id)
            tokens = tokens + [pad_token] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)


class Llama32Aggregator(TopAggregator):
    """PyTorch Llama 3.2 1B Aggregator for horizontal tensor parallelism"""

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
        self.enable_swapping = getattr(self.config.hyperparameters, 'enable_swapping', False)
        self.lr = self.config.hyperparameters.learning_rate
        self.rounds = self.config.hyperparameters.rounds
        self.ckpt_dir = getattr(self.config.hyperparameters, 'ckpt_dir', '../')
        self.max_seq_len = getattr(self.config.hyperparameters, 'max_seq_len', 512)
        self.batch_size = getattr(self.config.hyperparameters, 'batch_size', 4)
        
        # Initialize experiment results tracking
        self.experiment_results = []
        self.experiment_key = (self.world_size, self.lr, self.enable_swapping, self.rounds, seed)
        
        tokenizer_path = Path(os.path.join(self.ckpt_dir, "tokenizer.model"))
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def _save_experiment_results(self):
        """Save experiment results to a pickle file."""
        if not self.experiment_results:
            return
            
        results_file = "experiment_results_llama32.pkl"
        results_dict = {}
        
        # Load existing results if file exists
        if os.path.exists(results_file):
            try:
                with open(results_file, 'rb') as f:
                    results_dict = pickle.load(f)
            except Exception as e:
                logger.error(f"Could not load existing results: {e}")
        
        # Add current experiment results
        results_dict[self.experiment_key] = self.experiment_results

        # Duplicate res for swap=True for ws=1
        if self.world_size == 1 and not self.enable_swapping:
            no_swap_key = list(self.experiment_key)
            no_swap_key[2] = True
            results_dict[tuple(no_swap_key)] = self.experiment_results

        # Save updated results
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(results_dict, f)
            logger.info(f"Saved experiment results to {results_file}")
        except Exception as e:
            logger.error(f"Could not save experiment results: {e}")

    def _save_model_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = f"../checkpoints/llama32_3b_ws{self.world_size}_lr{self.lr}_round{self._round}"

        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save the state dict directly (compatible with generation.py loading)
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            "consolidated.00.pth"
        )
        
        # Save state dict in the same format as original checkpoints
        torch.save(self.model.state_dict(), checkpoint_path)
        
        # Also save metadata separately for experiment tracking
        metadata_path = os.path.join(
            checkpoint_dir,
            f"metadata_ws{self.world_size}_lr{self.lr}_round{self._round}.pkl"
        )
        
        metadata = {
            'experiment_key': self.experiment_key,
            'round': self._round,
            'config': self.config,
            'world_size': self.world_size,
            'learning_rate': self.lr,
            'enable_swapping': self.enable_swapping
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Copy params.json and tokenizer.model to the finetuned directory
        import shutil
        src_params = os.path.join(self.ckpt_dir, "params.json")
        src_tokenizer = os.path.join(self.ckpt_dir, "tokenizer.model")
        dst_params = os.path.join(checkpoint_dir, "params.json")
        dst_tokenizer = os.path.join(checkpoint_dir, "tokenizer.model")
        
        if os.path.exists(src_params):
            shutil.copy2(src_params, dst_params)
        if os.path.exists(src_tokenizer):
            shutil.copy2(src_tokenizer, dst_tokenizer)
        
        logger.info(f"Saved model checkpoint to {checkpoint_path}")
        logger.info(f"Saved experiment metadata to {metadata_path}")

    def initialize(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            logger.info("Aggregator using GPU 0")
        else:
            self.device = torch.device("cpu")
            logger.info("Aggregator using CPU (no GPUs available)")

        # Load model parameters
        params_path = os.path.join(self.ckpt_dir, "params.json")
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        params['max_seq_len'] = self.max_seq_len
        
        model_args = ModelArgs(**params)
        self.model = Transformer(model_args).to(self.device)
        
        
        self._load_pretrained_weights()

        # Evaluate baseline before training
        self.evaluate(baseline=True)

    def _load_pretrained_weights(self):
        """Load pretrained weights from checkpoint."""
        ckpt_path = os.path.join(self.ckpt_dir, "consolidated.00.pth")
        
        if not os.path.exists(ckpt_path):
            logger.error(f"Pretrained checkpoint not found at {ckpt_path}")
            raise RuntimeError(f"Pretrained checkpoint not found at {ckpt_path}")
        
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            model_dict = self.model.state_dict()
            adapted_dict = {}
            
            for name, param in checkpoint.items():
                if name in model_dict:
                    if param.shape == model_dict[name].shape:
                        adapted_dict[name] = param
                    else:
                        logger.error(f"Shape mismatch for {name}: {param.shape} vs {model_dict[name].shape}")
                        raise RuntimeError(f"Shape mismatch for {name}: {param.shape} vs {model_dict[name].shape}")
                else:
                    logger.error(f"Key {name} not found in model")
                    raise RuntimeError(f"Key {name} not found in model")
            
            # Load adapted weights
            model_dict.update(adapted_dict)
            self.model.load_state_dict(model_dict, strict=False)
            
            logger.info("Loaded pretrained weights successfully")
            
        except Exception as e:
            logger.error(f"Could not load pretrained weights: {e}")
            raise RuntimeError(f"Could not load pretrained weights: {e}")

    def load_data(self) -> None:
        # Set generator for reproducible data loading
        g = torch.Generator()
        g.manual_seed(getattr(self.config.hyperparameters, 'seed', 42))
        
        # For evaluation, create a small synthetic dataset
        # In practice, you would load your evaluation text data here
        eval_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world, this is a test sentence for evaluation.",
            "Machine learning is revolutionizing the world of artificial intelligence.",
            "Federated learning enables privacy-preserving distributed training.",
        ]
        
        # Create temporary evaluation file
        eval_file = "temp_eval.txt"
        with open(eval_file, 'w') as f:
            for text in eval_texts:
                f.write(text + "\n")
        
        eval_dataset = TextDataset(eval_file, self.tokenizer, self.max_seq_len)
        
        self.test_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(
                getattr(self.config.hyperparameters, 'seed', 42) + worker_id
            )
        )

        # Store data into dataset for analysis
        self.dataset = Dataset(dataloader=self.test_loader)
        
        # Clean up temporary file
        if os.path.exists(eval_file):
            os.remove(eval_file)

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass

    def evaluate(self, baseline=False) -> None:
        """Evaluate (test) a model."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                tokens = batch.to(self.device)
                
                # Create input and target sequences
                input_tokens = tokens[:, :-1]
                target_tokens = tokens[:, 1:]
                
                # Forward pass
                logits = self.model(input_tokens)
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1),
                    ignore_index=getattr(self.tokenizer, 'pad_id', self.tokenizer.eos_id)
                )
                
                # Calculate perplexity
                valid_tokens = (target_tokens != getattr(self.tokenizer, 'pad_id', self.tokenizer.eos_id)).sum()
                total_loss += loss.item() * valid_tokens.item()
                total_tokens += valid_tokens.item()

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Store result for pickle saving
        self.experiment_results.append((self._round if not baseline else 0, avg_loss, perplexity))

        # Write results to a file
        with open(f'eval_res_llama32_ws{self.world_size}_r{self._rounds}.txt', 'a') as f:
            f.write(f"Round: {self._round if not baseline else 0}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}\n")
        
        logger.info(f"Evaluated round {self._round if not baseline else 0}: Loss={avg_loss:.4f}, Perplexity={perplexity:.4f}")
        
        # Update metrics for model registry
        self.update_metrics({
            'test-loss': avg_loss,
            'perplexity': perplexity
        })
        
        # Save experiment results and checkpoint if this is the final round
        if self._round == self._rounds:
            self._save_experiment_results()
            self._save_model_checkpoint()

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

            logger.info(f"{end}'s parameters trained with {count} samples")

            if weights is not None and count > 0:
                count_total += count
                w_received += 1
                trainers_weights[self.trainer_rank[end]] = weights

            individual_count = count

        # Concatenate weights from all trainers
        if w_received == self.world_size:
            combined_weights = self._concatenate_weights(trainers_weights)
            tres = TrainResult(combined_weights, count_total)
            self.cache["concat"] = tres

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
            logger.error(f"Channel with tag {tag} not found")
            return

        # This call waits for at least one peer to join this channel
        channel.await_join()

        while len(channel.all_ends()) < self.world_size:
            logger.info(f"Waiting for more trainers to join: {len(channel.all_ends())}/{self.world_size}")
            time.sleep(3)

        # Before distributing weights, update it from global model
        self._update_weights()

        selected_ends = channel.ends()
        datasampler_metadata = self.datasampler.get_metadata(self._round, selected_ends)

        # Swapping logic (only if enabled)
        if self.enable_swapping and self._round % 2 == 0 and self.trainer_count == self.world_size:
            logger.info(f"Performing trainer swapping for round {self._round}")
            for key in self.trainer_rank:
                self.trainer_rank[key] = (self.trainer_rank[key] + 1) % self.world_size

        # Send out global model parameters to trainers
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
                        temp, DeviceType.GPU
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
        """Slice weights for distributed training with tensor parallelism on Llama 3.2 3B."""
        sliced = {}
        
        for name, full_tensor in state_dict.items():
            if 'layers.' in name and ('attention.wq' in name or 'attention.wk' in name or 'attention.wv' in name):
                # Attention wq, wk, wv: Split output dimension (first dimension) headwise
                if 'attention.wq' in name:
                    # wq: [n_heads * head_dim, dim] -> split by heads on first dimension
                    n_heads = self.model.params.n_heads
                    heads_per_trainer = n_heads // world_size
                    head_dim = self.model.params.dim // n_heads
                    start_head = rank * heads_per_trainer
                    end_head = (rank + 1) * heads_per_trainer
                    start_idx = start_head * head_dim
                    end_idx = end_head * head_dim
                    sliced[name] = full_tensor[start_idx:end_idx, :].clone()
                elif 'attention.wk' in name or 'attention.wv' in name:
                    # wk, wv: [n_kv_heads * head_dim, dim] -> split by kv heads on first dimension
                    n_kv_heads = self.model.params.n_kv_heads
                    kv_heads_per_trainer = n_kv_heads // world_size
                    head_dim = self.model.params.dim // self.model.params.n_heads
                    start_head = rank * kv_heads_per_trainer
                    end_head = (rank + 1) * kv_heads_per_trainer
                    start_idx = start_head * head_dim
                    end_idx = end_head * head_dim
                    sliced[name] = full_tensor[start_idx:end_idx, :].clone()
                    
            elif 'layers.' in name and 'attention.wo' in name:
                # Attention wo: Split input dimension (second dimension) headwise
                n_heads = self.model.params.n_heads
                heads_per_trainer = n_heads // world_size
                head_dim = self.model.params.dim // n_heads
                start_head = rank * heads_per_trainer
                end_head = (rank + 1) * heads_per_trainer
                start_idx = start_head * head_dim
                end_idx = end_head * head_dim
                sliced[name] = full_tensor[:, start_idx:end_idx].clone()
                
            elif 'layers.' in name and 'feed_forward.w1' in name:
                # FFN w1: Split output dimension (first dimension) evenly
                hidden_dim = full_tensor.shape[0]
                hidden_per_trainer = hidden_dim // world_size
                start_idx = rank * hidden_per_trainer
                end_idx = (rank + 1) * hidden_per_trainer
                sliced[name] = full_tensor[start_idx:end_idx, :].clone()
                
            elif 'layers.' in name and 'feed_forward.w3' in name:
                # FFN w3: Split output dimension (first dimension) evenly
                hidden_dim = full_tensor.shape[0]
                hidden_per_trainer = hidden_dim // world_size
                start_idx = rank * hidden_per_trainer
                end_idx = (rank + 1) * hidden_per_trainer
                sliced[name] = full_tensor[start_idx:end_idx, :].clone()
                
            elif 'layers.' in name and 'feed_forward.w2' in name:
                # FFN w2: Split input dimension (second dimension) evenly
                hidden_dim = full_tensor.shape[1]
                hidden_per_trainer = hidden_dim // world_size
                start_idx = rank * hidden_per_trainer
                end_idx = (rank + 1) * hidden_per_trainer
                sliced[name] = full_tensor[:, start_idx:end_idx].clone()
                
            else:
                # Replicate other parameters (embeddings, norms, output layer)
                sliced[name] = full_tensor.clone()
                
        return sliced

    def _concatenate_weights(self, trainers_weights: List[Dict]) -> Dict:
        """
        Concatenate weights from trainers for Llama 3.2 1B with tensor parallelism.
        """
        concatenated = {}
        
        for name in trainers_weights[0].keys():
            if 'layers.' in name and ('attention.wq' in name or 'attention.wk' in name or 'attention.wv' in name):
                # Attention wq, wk, wv: Concatenate along output dimension (first dimension)
                weights = [w[name] for w in trainers_weights]
                concatenated[name] = torch.cat(weights, dim=0)
                
            elif 'layers.' in name and 'attention.wo' in name:
                # Attention wo: Concatenate along input dimension (second dimension)
                weights = [w[name] for w in trainers_weights]
                concatenated[name] = torch.cat(weights, dim=1)
                
            elif 'layers.' in name and ('feed_forward.w1' in name or 'feed_forward.w3' in name):
                # FFN w1, w3: Concatenate along output dimension (first dimension)
                weights = [w[name] for w in trainers_weights]
                concatenated[name] = torch.cat(weights, dim=0)
                
            elif 'layers.' in name and 'feed_forward.w2' in name:
                # FFN w2: Concatenate along input dimension (second dimension)
                weights = [w[name] for w in trainers_weights]
                concatenated[name] = torch.cat(weights, dim=1)
                
            else:
                # Average replicated parameters (embeddings, norms, output layer)
                weights = [w[name] for w in trainers_weights]
                concatenated[name] = torch.mean(torch.stack(weights), dim=0)
        
        return concatenated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Llama 3.2 1B Federated Learning Aggregator')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = Llama32Aggregator(config)
    a.compose() 
    a.run()
