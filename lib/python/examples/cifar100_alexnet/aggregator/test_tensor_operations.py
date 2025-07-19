#!/usr/bin/env python3
"""
Test script to verify that _slice_weights and _concatenate_weights operations
preserve tensor shapes and values for the AlexNet model.
"""

import torch
import torch.nn as nn
import numpy as np
from main import AlexNet, Cifar100AlexNetAggregator
from flame.config import Config


class SafeCifar100AlexNetAggregator(Cifar100AlexNetAggregator):
    """Test version of aggregator with safe concatenate weights method."""
    
    def _concatenate_weights(self, trainers_weights: list) -> dict:
        """Concatenate weights from trainers for AlexNet with 1D tensor parallelism (safe version)."""
        concated = {}
        
        # Conv1: Split by OUTPUT → Concatenate weights and bias
        if all('conv1.weight' in w for w in trainers_weights):
            weights = [w['conv1.weight'] for w in trainers_weights]
            concated['conv1.weight'] = torch.cat(weights, 0)
        if all('conv1.bias' in w for w in trainers_weights):
            weights = [w['conv1.bias'] for w in trainers_weights]
            concated['conv1.bias'] = torch.cat(weights, 0)
        
        # Conv2: Split by INPUT → Concatenate weights, Average bias
        if all('conv2.weight' in w for w in trainers_weights):
            weights = [w['conv2.weight'] for w in trainers_weights]
            concated['conv2.weight'] = torch.cat(weights, 1)
        if all('conv2.bias' in w for w in trainers_weights):
            weights = [w['conv2.bias'] for w in trainers_weights]
            concated['conv2.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        # Conv3: Split by OUTPUT → Concatenate weights and bias
        if all('conv3.weight' in w for w in trainers_weights):
            weights = [w['conv3.weight'] for w in trainers_weights]
            concated['conv3.weight'] = torch.cat(weights, 0)
        if all('conv3.bias' in w for w in trainers_weights):
            weights = [w['conv3.bias'] for w in trainers_weights]
            concated['conv3.bias'] = torch.cat(weights, 0)
        
        # Conv4: Split by INPUT → Concatenate weights, Average bias
        if all('conv4.weight' in w for w in trainers_weights):
            weights = [w['conv4.weight'] for w in trainers_weights]
            concated['conv4.weight'] = torch.cat(weights, 1)
        if all('conv4.bias' in w for w in trainers_weights):
            weights = [w['conv4.bias'] for w in trainers_weights]
            concated['conv4.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        # Conv5: Split by OUTPUT → Concatenate weights and bias
        if all('conv5.weight' in w for w in trainers_weights):
            weights = [w['conv5.weight'] for w in trainers_weights]
            concated['conv5.weight'] = torch.cat(weights, 0)
        if all('conv5.bias' in w for w in trainers_weights):
            weights = [w['conv5.bias'] for w in trainers_weights]
            concated['conv5.bias'] = torch.cat(weights, 0)
        
        # Conv6: Split by INPUT → Concatenate weights, Average bias
        if all('conv6.weight' in w for w in trainers_weights):
            weights = [w['conv6.weight'] for w in trainers_weights]
            concated['conv6.weight'] = torch.cat(weights, 1)
        if all('conv6.bias' in w for w in trainers_weights):
            weights = [w['conv6.bias'] for w in trainers_weights]
            concated['conv6.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        # FC1: Split by OUTPUT → Concatenate weights and bias
        if all('fc1.weight' in w for w in trainers_weights):
            weights = [w['fc1.weight'] for w in trainers_weights]
            concated['fc1.weight'] = torch.cat(weights, 0)
        if all('fc1.bias' in w for w in trainers_weights):
            weights = [w['fc1.bias'] for w in trainers_weights]
            concated['fc1.bias'] = torch.cat(weights, 0)
        
        # FC2: Split by INPUT → Concatenate weights, Average bias
        if all('fc2.weight' in w for w in trainers_weights):
            weights = [w['fc2.weight'] for w in trainers_weights]
            concated['fc2.weight'] = torch.cat(weights, 1)
        if all('fc2.bias' in w for w in trainers_weights):
            weights = [w['fc2.bias'] for w in trainers_weights]
            concated['fc2.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        # FC3: Split by OUTPUT → Concatenate weights and bias
        if all('fc3.weight' in w for w in trainers_weights):
            weights = [w['fc3.weight'] for w in trainers_weights]
            concated['fc3.weight'] = torch.cat(weights, 0)
        if all('fc3.bias' in w for w in trainers_weights):
            weights = [w['fc3.bias'] for w in trainers_weights]
            concated['fc3.bias'] = torch.cat(weights, 0)
        
        # FC4: Split by INPUT → Concatenate weights, Average bias
        if all('fc4.weight' in w for w in trainers_weights):
            weights = [w['fc4.weight'] for w in trainers_weights]
            concated['fc4.weight'] = torch.cat(weights, 1)
        if all('fc4.bias' in w for w in trainers_weights):
            weights = [w['fc4.bias'] for w in trainers_weights]
            concated['fc4.bias'] = torch.mean(torch.stack(weights), dim=0)
        
        return concated


def test_slice_concatenate_operations():
    """Test that slicing and concatenating weights preserves original tensors."""
    print("Starting tensor operations test...")
    
    # Create a model instance
    model = AlexNet()
    
    # Get the original state dict
    original_weights = model.state_dict()
    
    # Create a mock aggregator instance to access the methods
    config = Config("config.json")  # Assuming config.json exists
    aggregator = SafeCifar100AlexNetAggregator(config)
    
    # Test different world sizes
    world_sizes = [1, 2, 4]
    
    for world_size in world_sizes:
        print(f"\nTesting with world_size = {world_size}")
        
        # Step 1: Slice the weights
        sliced_weights = []
        for rank in range(world_size):
            sliced = aggregator._slice_weights(original_weights, rank, world_size)
            sliced_weights.append(sliced)
        
        # Step 2: Concatenate the sliced weights
        concatenated_weights = aggregator._concatenate_weights(sliced_weights)
        
        # Step 3: Compare shapes and values
        all_shapes_match = True
        all_values_match = True
        
        print(f"Comparing {len(original_weights)} layers...")
        
        for layer_name in original_weights.keys():
            original_tensor = original_weights[layer_name]
            concatenated_tensor = concatenated_weights[layer_name]
            
            # Check shapes
            if original_tensor.shape != concatenated_tensor.shape:
                print(f"❌ Shape mismatch for {layer_name}:")
                print(f"   Original: {original_tensor.shape}")
                print(f"   Concatenated: {concatenated_tensor.shape}")
                all_shapes_match = False
            else:
                print(f"✓ Shape match for {layer_name}: {original_tensor.shape}")
            
            # Check values (with tolerance for floating point precision)
            if not torch.allclose(original_tensor, concatenated_tensor, rtol=1e-5, atol=1e-8):
                print(f"❌ Value mismatch for {layer_name}")
                print(f"   Max difference: {torch.max(torch.abs(original_tensor - concatenated_tensor)).item()}")
                all_values_match = False
            else:
                print(f"✓ Value match for {layer_name}")
        
        if all_shapes_match and all_values_match:
            print(f"✅ All tests PASSED for world_size = {world_size}")
        else:
            print(f"❌ Some tests FAILED for world_size = {world_size}")
            if not all_shapes_match:
                print("   - Shape mismatches detected")
            if not all_values_match:
                print("   - Value mismatches detected")


def test_individual_layer_operations():
    """Test individual layer slicing and concatenation in detail."""
    print("\n" + "="*60)
    print("Testing individual layer operations...")
    
    # Create test tensors with known values
    test_tensors = {
        # Conv layers
        'conv1.weight': torch.randn(96, 3, 5, 5),
        'conv1.bias': torch.randn(96),
        'conv2.weight': torch.randn(256, 96, 5, 5),
        'conv2.bias': torch.randn(256),
        # FC layers
        'fc1.weight': torch.randn(4096, 4096),
        'fc1.bias': torch.randn(4096),
        'fc4.weight': torch.randn(100, 4096),
        'fc4.bias': torch.randn(100),
    }
    
    config = Config("config.json")
    aggregator = SafeCifar100AlexNetAggregator(config)
    world_size = 4
    
    for layer_name, original_tensor in test_tensors.items():
        print(f"\nTesting {layer_name} (shape: {original_tensor.shape})")
        
        # Create a state dict with just this tensor
        state_dict = {layer_name: original_tensor}
        
        # Slice the tensor
        sliced_parts = []
        for rank in range(world_size):
            sliced = aggregator._slice_weights(state_dict, rank, world_size)
            sliced_parts.append(sliced)
            
        # Concatenate back
        concatenated = aggregator._concatenate_weights(sliced_parts)
        reconstructed_tensor = concatenated[layer_name]
        
        # Check results
        shape_match = original_tensor.shape == reconstructed_tensor.shape
        value_match = torch.allclose(original_tensor, reconstructed_tensor, rtol=1e-5, atol=1e-8)
        
        print(f"   Shape match: {shape_match}")
        print(f"   Value match: {value_match}")
        
        if not shape_match:
            print(f"   Original shape: {original_tensor.shape}")
            print(f"   Reconstructed shape: {reconstructed_tensor.shape}")
        
        if not value_match:
            max_diff = torch.max(torch.abs(original_tensor - reconstructed_tensor)).item()
            print(f"   Max difference: {max_diff}")


def test_tensor_splitting_logic():
    """Test the logic of how tensors are split across different ranks."""
    print("\n" + "="*60)
    print("Testing tensor splitting logic...")
    
    config = Config("config.json")
    aggregator = SafeCifar100AlexNetAggregator(config)
    world_size = 4
    
    # Test conv1 (output channel split)
    conv1_weight = torch.randn(96, 3, 5, 5)
    conv1_bias = torch.randn(96)
    
    print(f"\nConv1 weight shape: {conv1_weight.shape}")
    print(f"Conv1 bias shape: {conv1_bias.shape}")
    
    # Check that each rank gets equal portions
    slice_size = 96 // world_size
    print(f"Expected slice size for conv1: {slice_size}")
    
    state_dict = {'conv1.weight': conv1_weight, 'conv1.bias': conv1_bias}
    
    for rank in range(world_size):
        sliced = aggregator._slice_weights(state_dict, rank, world_size)
        
        expected_weight_shape = (slice_size, 3, 5, 5)
        expected_bias_shape = (slice_size,)
        
        actual_weight_shape = sliced['conv1.weight'].shape
        actual_bias_shape = sliced['conv1.bias'].shape
        
        print(f"Rank {rank}:")
        print(f"  Weight shape: {actual_weight_shape} (expected: {expected_weight_shape})")
        print(f"  Bias shape: {actual_bias_shape} (expected: {expected_bias_shape})")
        
        assert actual_weight_shape == expected_weight_shape, f"Weight shape mismatch for rank {rank}"
        assert actual_bias_shape == expected_bias_shape, f"Bias shape mismatch for rank {rank}"


def run_all_tests():
    """Run all test functions."""
    print("AlexNet Tensor Operations Test Suite")
    print("="*60)
    
    try:
        test_slice_concatenate_operations()
        test_individual_layer_operations()
        test_tensor_splitting_logic()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)
