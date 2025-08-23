#!/usr/bin/env python3
"""
Test script to verify _slice_weights and _concatenate_weights methods
for the Llama 3.2 1B Transformer model.

This script checks:
1. Weight slicing and concatenation preserve original values
2. Shape consistency across different world sizes
3. Proper tensor parallelism implementation
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_no_fairscale import Transformer
from args import ModelArgs


class WeightOperationTester:
    """Test class for weight slicing and concatenation operations."""
    
    def __init__(self, model_params_path: str = None, use_cpu: bool = False):
        """Initialize the tester with model parameters."""
        self.device = torch.device("cpu") if use_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model parameters or use defaults
        if model_params_path and os.path.exists(model_params_path):
            with open(model_params_path, 'r') as f:
                params = json.load(f)
        else:
            # Default Llama 3.2 1B parameters for testing
            params = {
                "dim": 2048,
                "n_layers": 16,
                "n_heads": 32,
                "n_kv_heads": 8,
                "vocab_size": 128256,
                "norm_eps": 1e-5,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
                "max_seq_len": 512,
                "max_batch_size": 8
            }
        
        self.model_args = ModelArgs(**params)
        self.model = Transformer(self.model_args).to(self.device)
        
        # Initialize model with random weights for testing
        self._init_random_weights()
        
        print(f"Initialized model with parameters:")
        print(f"  - dim: {self.model_args.dim}")
        print(f"  - n_layers: {self.model_args.n_layers}")
        print(f"  - n_heads: {self.model_args.n_heads}")
        print(f"  - n_kv_heads: {self.model_args.n_kv_heads}")
        print(f"  - vocab_size: {self.model_args.vocab_size}")
        print(f"  - device: {self.device}")
    
    def _init_random_weights(self):
        """Initialize model with random weights for consistent testing."""
        torch.manual_seed(42)
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.1)
    
    def _slice_weights(self, state_dict: Dict, rank: int, world_size: int) -> Dict:
        """
        Slice weights for distributed training with tensor parallelism on Llama 3.2 3B.
        This is a copy of the method from main.py for testing.
        """
        sliced = {}
        
        for name, full_tensor in state_dict.items():
            if 'layers.' in name and ('attention.wq' in name or 'attention.wk' in name or 'attention.wv' in name):
                # Attention wq, wk, wv: Split output dimension (first dimension) headwise
                if 'attention.wq' in name:
                    # wq: [n_heads * head_dim, dim] -> split by heads on first dimension
                    n_heads = self.model_args.n_heads
                    heads_per_trainer = n_heads // world_size
                    head_dim = self.model_args.dim // n_heads
                    start_head = rank * heads_per_trainer
                    end_head = (rank + 1) * heads_per_trainer
                    start_idx = start_head * head_dim
                    end_idx = end_head * head_dim
                    sliced[name] = full_tensor[start_idx:end_idx, :].clone()
                elif 'attention.wk' in name or 'attention.wv' in name:
                    # wk, wv: [n_kv_heads * head_dim, dim] -> split by kv heads on first dimension
                    n_kv_heads = self.model_args.n_kv_heads
                    kv_heads_per_trainer = n_kv_heads // world_size
                    head_dim = self.model_args.dim // self.model_args.n_heads
                    start_head = rank * kv_heads_per_trainer
                    end_head = (rank + 1) * kv_heads_per_trainer
                    start_idx = start_head * head_dim
                    end_idx = end_head * head_dim
                    sliced[name] = full_tensor[start_idx:end_idx, :].clone()
                    
            elif 'layers.' in name and 'attention.wo' in name:
                # Attention wo: Split input dimension (second dimension) headwise
                n_heads = self.model_args.n_heads
                heads_per_trainer = n_heads // world_size
                head_dim = self.model_args.dim // n_heads
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
        This is a copy of the method from main.py for testing.
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
    
    def test_shape_consistency(self, world_size: int) -> bool:
        """Test that slicing and concatenation preserve shapes."""
        print(f"\n=== Testing shape consistency for world_size={world_size} ===")
        
        original_state_dict = self.model.state_dict()
        
        # Check divisibility constraints
        if self.model_args.n_heads % world_size != 0:
            print(f"SKIP: n_heads ({self.model_args.n_heads}) not divisible by world_size ({world_size})")
            return False
        
        if self.model_args.n_kv_heads % world_size != 0:
            print(f"SKIP: n_kv_heads ({self.model_args.n_kv_heads}) not divisible by world_size ({world_size})")
            return False
        
        # Slice weights for each trainer
        sliced_weights = []
        for rank in range(world_size):
            sliced = self._slice_weights(original_state_dict, rank, world_size)
            sliced_weights.append(sliced)
        
        # Concatenate weights back
        reconstructed = self._concatenate_weights(sliced_weights)
        
        # Check shapes
        shape_match = True
        for name, original_tensor in original_state_dict.items():
            if name not in reconstructed:
                print(f"ERROR: Missing parameter {name} in reconstructed weights")
                shape_match = False
                continue
            
            if original_tensor.shape != reconstructed[name].shape:
                print(f"ERROR: Shape mismatch for {name}")
                print(f"  Original: {original_tensor.shape}")
                print(f"  Reconstructed: {reconstructed[name].shape}")
                shape_match = False
        
        if shape_match:
            print("✓ All shapes match")
        
        return shape_match
    
    def test_value_preservation(self, world_size: int, tolerance: float = 1e-6) -> bool:
        """Test that slicing and concatenation preserve values."""
        print(f"\n=== Testing value preservation for world_size={world_size} ===")
        
        original_state_dict = self.model.state_dict()
        
        # Check divisibility constraints
        if self.model_args.n_heads % world_size != 0 or self.model_args.n_kv_heads % world_size != 0:
            print(f"SKIP: Heads not divisible by world_size ({world_size})")
            return False
        
        # Slice weights for each trainer
        sliced_weights = []
        for rank in range(world_size):
            sliced = self._slice_weights(original_state_dict, rank, world_size)
            sliced_weights.append(sliced)
        
        # Concatenate weights back
        reconstructed = self._concatenate_weights(sliced_weights)
        
        # Check values
        value_match = True
        max_diff = 0.0
        
        for name, original_tensor in original_state_dict.items():
            if name not in reconstructed:
                continue
            
            # For replicated parameters, we expect averaging, so compare differently
            if not ('layers.' in name and any(x in name for x in ['attention.', 'feed_forward.'])):
                # These are replicated parameters (embeddings, norms, output)
                # They should be identical to the original (since all slices are the same)
                diff = torch.abs(original_tensor - reconstructed[name])
            else:
                # These are split parameters, should be exactly reconstructed
                diff = torch.abs(original_tensor - reconstructed[name])
            
            max_param_diff = diff.max().item()
            max_diff = max(max_diff, max_param_diff)
            
            if max_param_diff > tolerance:
                print(f"ERROR: Value mismatch for {name}")
                print(f"  Max difference: {max_param_diff}")
                print(f"  Original range: [{original_tensor.min().item():.6f}, {original_tensor.max().item():.6f}]")
                print(f"  Reconstructed range: [{reconstructed[name].min().item():.6f}, {reconstructed[name].max().item():.6f}]")
                value_match = False
        
        if value_match:
            print(f"✓ All values preserved (max diff: {max_diff:.2e})")
        
        return value_match
    
    def test_slice_shapes(self, world_size: int):
        """Test individual slice shapes."""
        print(f"\n=== Testing slice shapes for world_size={world_size} ===")
        
        original_state_dict = self.model.state_dict()
        
        # Check divisibility constraints
        if self.model_args.n_heads % world_size != 0 or self.model_args.n_kv_heads % world_size != 0:
            print(f"SKIP: Heads not divisible by world_size ({world_size})")
            return
        
        for rank in range(world_size):
            print(f"\nTrainer rank {rank}:")
            sliced = self._slice_weights(original_state_dict, rank, world_size)
            
            # Check attention layer shapes
            for name, tensor in sliced.items():
                if 'layers.0.' in name:  # Only check first layer for brevity
                    if 'attention.wq' in name:
                        expected_dim = self.model_args.dim // self.model_args.n_heads * (self.model_args.n_heads // world_size)
                        print(f"  {name}: {tensor.shape} (expected first dim: {expected_dim})")
                    elif 'attention.wk' in name or 'attention.wv' in name:
                        expected_dim = self.model_args.dim // self.model_args.n_heads * (self.model_args.n_kv_heads // world_size)
                        print(f"  {name}: {tensor.shape} (expected first dim: {expected_dim})")
                    elif 'attention.wo' in name:
                        expected_dim = self.model_args.dim // self.model_args.n_heads * (self.model_args.n_heads // world_size)
                        print(f"  {name}: {tensor.shape} (expected second dim: {expected_dim})")
                    elif 'feed_forward.' in name:
                        print(f"  {name}: {tensor.shape}")
    
    def run_comprehensive_test(self):
        """Run all tests for different world sizes."""
        print("=" * 80)
        print("COMPREHENSIVE WEIGHT OPERATION TEST")
        print("=" * 80)
        
        # Test different world sizes
        world_sizes = [1, 2, 4]
        results = {}
        
        for ws in world_sizes:
            print(f"\n{'='*20} Testing World Size {ws} {'='*20}")
            
            # Check if this world size is valid
            if (self.model_args.n_heads % ws != 0 or 
                self.model_args.n_kv_heads % ws != 0):
                print(f"SKIP: World size {ws} not compatible with model architecture")
                results[ws] = "INCOMPATIBLE"
                continue
            
            # Test slice shapes
            self.test_slice_shapes(ws)
            
            # Test shape consistency
            shape_ok = self.test_shape_consistency(ws)
            
            # Test value preservation
            value_ok = self.test_value_preservation(ws)
            
            # Store results
            if shape_ok and value_ok:
                results[ws] = "PASS"
            else:
                results[ws] = "FAIL"
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        for ws, result in results.items():
            status_symbol = "✓" if result == "PASS" else "✗" if result == "FAIL" else "~"
            print(f"World Size {ws:2d}: {status_symbol} {result}")
        
        return results


def main():
    """Main test function."""
    # Try to load model parameters from the checkpoint directory
    ckpt_dir = "../checkpoints"
    params_path = os.path.join(ckpt_dir, "params.json")
    
    if not os.path.exists(params_path):
        params_path = None
        print("Using default model parameters (params.json not found)")
    
    # Create tester and run tests
    tester = WeightOperationTester(params_path)
    results = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    if all(r in ["PASS", "INCOMPATIBLE"] for r in results.values()):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
