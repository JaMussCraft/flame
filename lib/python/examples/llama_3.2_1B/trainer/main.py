"""Llama 3.2 1B horizontal FL trainer for PyTorch.

This example demonstrates distributed training of Llama 3.2 1B model
using federated learning with horizontal tensor parallelism.
"""

import argparse
import json
import logging
import os
import random
import sys
import math
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, Dataset as TorchDataset

from flame.config import Config
from flame.mode.horizontal.syncfl.trainer import Trainer

# Import from aggregator directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aggregator'))
from model_no_fairscale import RMSNorm, apply_rotary_emb, repeat_kv, precompute_freqs_cis, reshape_for_broadcast
from args import ModelArgs
from tokenizer import Tokenizer

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


class TextDataset(TorchDataset):
    """Text dataset for training."""
    
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_length: int = 512, rank: int = 0, world_size: int = 1):
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
        
        # Simple data partitioning - each trainer gets a subset
        samples_per_trainer = len(self.texts) // world_size
        start_idx = rank * samples_per_trainer
        end_idx = start_idx + samples_per_trainer
        self.texts = self.texts[start_idx:end_idx]
        
        logger.info(f"Trainer {rank} loaded {len(self.texts)} text samples")
    
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


class HorizontallySplitAttention(nn.Module):
    """Attention with horizontal tensor parallelism."""
    
    def __init__(self, args: ModelArgs, rank: int, world_size: int):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.rank = rank
        self.world_size = world_size
        
        self.heads_per_trainer = self.n_heads // world_size
        self.kv_heads_per_trainer = self.n_kv_heads // world_size
        
        # Split attention heads across trainers
        self.wq = nn.Linear(args.dim, self.heads_per_trainer * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.kv_heads_per_trainer * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.kv_heads_per_trainer * self.head_dim, bias=False)
        self.wo = nn.Linear(self.heads_per_trainer * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.kv_heads_per_trainer,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.kv_heads_per_trainer,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.heads_per_trainer, self.head_dim)
        xk = xk.view(bsz, seqlen, self.kv_heads_per_trainer, self.head_dim)
        xv = xv.view(bsz, seqlen, self.kv_heads_per_trainer, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads for this trainer
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)  # (bs, heads_per_trainer, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, heads_per_trainer, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, heads_per_trainer, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, heads_per_trainer, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, heads_per_trainer, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class HorizontallySplitFeedForward(nn.Module):
    """Feed-forward network with horizontal tensor parallelism."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        rank: int,
        world_size: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.rank = rank
        self.world_size = world_size
        self.hidden_per_trainer = hidden_dim // world_size

        # Split FFN weights across trainers
        self.w1 = nn.Linear(dim, self.hidden_per_trainer, bias=False)
        self.w2 = nn.Linear(self.hidden_per_trainer, dim, bias=False)
        self.w3 = nn.Linear(dim, self.hidden_per_trainer, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class HorizontallySplitTransformerBlock(nn.Module):
    """Transformer block with horizontal tensor parallelism."""
    
    def __init__(self, layer_id: int, args: ModelArgs, rank: int, world_size: int):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = HorizontallySplitAttention(args, rank, world_size)
        self.feed_forward = HorizontallySplitFeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            rank=rank,
            world_size=world_size,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class HorizontallySplitTransformer(nn.Module):
    """Transformer with horizontal tensor parallelism."""
    
    def __init__(self, params: ModelArgs, rank: int, world_size: int):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.rank = rank
        self.world_size = world_size

        # Replicated layers (not split)
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(HorizontallySplitTransformerBlock(layer_id, params, rank, world_size))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            # Handle MPS device bug
            if mask.device.type == torch.device("mps").type:
                mask = torch.nan_to_num(mask, nan=0.0)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence.
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


class HorizontalSplitTrainer(Trainer):
    """Llama 3.2 1B Horizontal Split Trainer."""
    
    def __init__(self, config: Config):
        # Set seed for reproducibility
        seed = getattr(config.hyperparameters, 'seed', 42)
        set_seed(seed)
        
        self.config = config
        self.rank = self.config.hyperparameters.rank
        self.world_size = self.config.hyperparameters.world_size
        self.dataset_size = 0
        self.epochs = config.hyperparameters.epochs
        self.batch_size = config.hyperparameters.batch_size
        self.lr = self.config.hyperparameters.learning_rate
        self.weight_decay = getattr(config.hyperparameters, 'weight_decay', 0.01)
        self.max_seq_len = getattr(config.hyperparameters, 'max_seq_len', 512)
        self.max_batch_size = getattr(config.hyperparameters, 'max_batch_size', 8)
        self.ckpt_dir = getattr(config.hyperparameters, 'ckpt_dir', '../llama3')
        self.data_path = getattr(config.hyperparameters, 'data_path', 'train_data.txt')
        
        self.train_loader = None
        self.model = None

        # Initialize tokenizer
        tokenizer_path = Path(os.path.join(self.ckpt_dir, "tokenizer.model"))
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def initialize(self) -> None:
        """Initialize trainer."""
        # Auto-detect available GPUs and assign based on rank
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_id = (self.rank % num_gpus) + 1 # to avoid sharing GPU 0 with aggregator
            self.device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
            logger.info(f"Trainer rank {self.rank} using GPU {gpu_id}")
        else:
            self.device = torch.device("cpu")
            logger.info(f"Trainer rank {self.rank} using CPU (no GPUs available)")

        # Load model parameters
        params_path = os.path.join(self.ckpt_dir, "params.json")
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Update params with config values
        params['max_seq_len'] = self.max_seq_len
        params['max_batch_size'] = self.max_batch_size
        
        model_args = ModelArgs(**params)
        self.model = HorizontallySplitTransformer(model_args, self.rank, self.world_size).to(self.device)
        
        
        logger.info(f"Initialized Llama 3.2 1B trainer {self.rank+1}/{self.world_size}")

    def load_data(self):
        """Load training data."""
        # Create a simple training dataset if it doesn't exist
        if not os.path.exists(self.data_path):
            logger.info(f"Creating sample training data at {self.data_path}")
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming the world with artificial intelligence.",
                "Federated learning enables privacy-preserving distributed training across multiple devices.",
                "Large language models like Llama are revolutionizing natural language processing.",
                "Transformer architectures have become the foundation for modern AI systems.",
                "PyTorch provides a flexible framework for deep learning research and development.",
                "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
                "Neural networks learn complex patterns through backpropagation and gradient descent.",
                "Data preprocessing is crucial for training effective machine learning models.",
                "Cross-entropy loss is commonly used for classification tasks in deep learning.",
            ]
            
            with open(self.data_path, 'w') as f:
                for text in sample_texts:
                    f.write(text + "\n")
        
        # Set generator for reproducible data loading
        g = torch.Generator()
        g.manual_seed(getattr(self.config.hyperparameters, 'seed', 42))
        
        dataset = TextDataset(
            self.data_path, 
            self.tokenizer, 
            self.max_seq_len, 
            self.rank, 
            self.world_size
        )
        
        self.train_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(
                getattr(self.config.hyperparameters, 'seed', 42) + worker_id
            )
        )
        
        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(dataset)
        logger.info(f"Trainer {self.rank} loaded {self.dataset_size} training samples")

    def train(self) -> None:
        """Train the model."""
        logger.info("Beginning training!")
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        for epoch in range(1, self.epochs + 1):
            print(f"Training epoch {epoch}")
            self._train_epoch(epoch)

    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            tokens = batch.to(self.device)
            
            # Create input and target sequences for language modeling
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            
            self.optimizer.zero_grad()
            
            logits = self.model(input_tokens, start_pos=0)
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
                ignore_index=getattr(self.tokenizer, 'pad_id', self.tokenizer.eos_id)
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                logger.info(f"Rank {self.rank}, Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                           f"Loss: {loss.item():.6f}")
        

    def evaluate(self) -> None:
        """Evaluate the model."""
        # Implement evaluation if needed
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Llama 3.2 1B Federated Learning Trainer')
    parser.add_argument("config", nargs="?", default="config.json")
    args = parser.parse_args()
    config = Config(args.config)
    t = HorizontalSplitTrainer(config)
    t.compose()
    t.run()
