#!/usr/bin/env python3
"""
Simple test script to verify ResNet34 tensor parallelism implementation.
This tests the model architecture and tensor splitting without running full FL.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add the project path
sys.path.append('/home/cc/flame/lib/python')

from trainer.main import HorizontallySplitResNet34
from aggregator.main import ResNet34

def test_model_shapes():
    """Test that split and full models have compatible shapes."""
    print("Testing model architectures...")
    
    # Test parameters
    batch_size = 4
    world_size = 2
    
    # Create models
    full_model = ResNet34(num_classes=100)
    split_model_0 = HorizontallySplitResNet34(rank=0, world_size=world_size, num_classes=100)
    split_model_1 = HorizontallySplitResNet34(rank=1, world_size=world_size, num_classes=100)
    
    # Create input (CIFAR-100 scaled to ImageNet size)
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Test forward passes
    with torch.no_grad():
        full_output = full_model(x)
        split_output_0 = split_model_0(x)
        split_output_1 = split_model_1(x)
    
    print(f"Full model output shape: {full_output.shape}")
    print(f"Split model 0 output shape: {split_output_0.shape}")
    print(f"Split model 1 output shape: {split_output_1.shape}")
    
    # All outputs should have the same shape
    assert full_output.shape == split_output_0.shape == split_output_1.shape, \
        "Output shapes don't match!"
    
    print("✓ Model shapes are correct!")

def test_parameter_splitting():
    """Test that parameters are properly split."""
    print("\nTesting parameter splitting...")
    
    world_size = 2
    full_model = ResNet34(num_classes=100)
    split_model_0 = HorizontallySplitResNet34(rank=0, world_size=world_size, num_classes=100)
    split_model_1 = HorizontallySplitResNet34(rank=1, world_size=world_size, num_classes=100)
    
    # Count parameters
    full_params = sum(p.numel() for p in full_model.parameters())
    split_params_0 = sum(p.numel() for p in split_model_0.parameters())
    split_params_1 = sum(p.numel() for p in split_model_1.parameters())
    
    print(f"Full model parameters: {full_params:,}")
    print(f"Split model 0 parameters: {split_params_0:,}")
    print(f"Split model 1 parameters: {split_params_1:,}")
    
    # Split models should have fewer parameters
    assert split_params_0 < full_params, "Split model should have fewer parameters!"
    assert split_params_1 < full_params, "Split model should have fewer parameters!"
    
    # Both split models should have roughly the same number of parameters
    param_diff = abs(split_params_0 - split_params_1)
    param_avg = (split_params_0 + split_params_1) / 2
    relative_diff = param_diff / param_avg
    
    print(f"Parameter difference: {param_diff:,} ({relative_diff:.2%})")
    
    # Should be balanced (within 5% difference)
    assert relative_diff < 0.05, "Parameter split is not balanced!"
    
    print("✓ Parameter splitting is correct!")

def test_key_layers():
    """Test specific layer configurations."""
    print("\nTesting key layer configurations...")
    
    world_size = 2
    split_model = HorizontallySplitResNet34(rank=0, world_size=world_size, num_classes=100)
    
    # Test initial conv layer
    initial_conv = split_model.conv1
    assert initial_conv.out_channels == 64 // world_size, \
        f"Initial conv should have {64//world_size} output channels, got {initial_conv.out_channels}"
    
    # Test BasicBlock conv1 and conv2
    first_block = split_model.layer1[0]
    assert first_block.conv1.out_channels == 64 // world_size, \
        f"BasicBlock conv1 should have {64//world_size} output channels"
    assert first_block.conv2.out_channels == 64, \
        f"BasicBlock conv2 should have 64 output channels (full), got {first_block.conv2.out_channels}"
    
    # Test final FC layer
    fc_layer = split_model.fc
    assert fc_layer.in_features == 512, \
        f"FC layer should have 512 input features (full), got {fc_layer.in_features}"
    assert fc_layer.out_features == 100, \
        f"FC layer should have 100 output features, got {fc_layer.out_features}"
    
    print("✓ Key layer configurations are correct!")

def test_tensor_flow():
    """Test that tensors flow correctly through the network."""
    print("\nTesting tensor flow...")
    
    batch_size = 2
    world_size = 2
    split_model = HorizontallySplitResNet34(rank=0, world_size=world_size, num_classes=100)
    
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Test intermediate shapes
    with torch.no_grad():
        # Initial layers
        x1 = split_model.conv1(x)  # Should be [batch, 32, 112, 112] for rank 0
        assert x1.shape[1] == 64 // world_size, f"After conv1: expected {64//world_size} channels, got {x1.shape[1]}"
        
        x1 = split_model.bn1(x1)
        x1 = split_model.relu(x1)
        x1 = split_model.maxpool(x1)  # [batch, 32, 56, 56]
        
        # First BasicBlock
        first_block = split_model.layer1[0]
        identity = first_block.shortcut(x1)  # Should be [batch, 64, 56, 56] (full)
        
        out = first_block.conv1(x1)  # [batch, 32, 56, 56] (split)
        out = first_block.bn1(out)
        out = first_block.relu(out)
        out = first_block.conv2(out)  # [batch, 64, 56, 56] (full)
        
        assert out.shape[1] == 64, f"After BasicBlock conv2: expected 64 channels, got {out.shape[1]}"
        assert identity.shape == out.shape, f"Identity and main path shapes don't match: {identity.shape} vs {out.shape}"
    
    print("✓ Tensor flow is correct!")

if __name__ == "__main__":
    print("=" * 50)
    print("ResNet34 Tensor Parallelism Test")
    print("=" * 50)
    
    try:
        test_model_shapes()
        test_parameter_splitting()
        test_key_layers()
        test_tensor_flow()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Implementation is correct.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
