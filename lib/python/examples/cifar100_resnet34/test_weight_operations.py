#!/usr/bin/env python3
"""
Fixed test script to verify _slice_weights and _concatenate_weights operations
for ResNet34 model in horizontal federated learning.

This script tests:
1. Shape preservation during slice and concatenate operations
2. Value preservation for split parameters
3. Averaging behavior for replicated parameters
4. Complete round-trip: original -> slice -> concatenate -> compare
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from copy import deepcopy

# Add the aggregator directory to path to import the classes
sys.path.append(os.path.join(os.path.dirname(__file__), 'aggregator'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trainer'))

# Import from aggregator
import aggregator.main as agg_main
# Import from trainer  
import trainer.main as trainer_main


def set_seed(seed=42):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class WeightOperationsTester:
    """Test class for weight slicing and concatenation operations."""
    
    def __init__(self, world_size=2, num_classes=100):
        self.world_size = world_size
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a dummy aggregator to access the methods
        self.aggregator = self._create_dummy_aggregator()
        
        # Create models
        self.full_model = agg_main.ResNet34(num_classes=num_classes).to(self.device)
        
    def _create_dummy_aggregator(self):
        """Create a minimal aggregator instance for accessing methods."""
        class DummyConfig:
            class Hyperparameters:
                seed = 42
                world_size = self.world_size
                learning_rate = 0.01
                rounds = 1
                pretrain = False
                
            hyperparameters = Hyperparameters()
        
        config = DummyConfig()
        aggregator = agg_main.Cifar100ResNet34Aggregator(config)
        aggregator.world_size = self.world_size
        return aggregator
    
    def test_parameter_coverage(self):
        """Test that all model parameters are handled correctly."""
        print("Testing parameter coverage...")
        
        original_weights = self.full_model.state_dict()
        sliced_weights = self.aggregator._slice_weights(original_weights, 0, self.world_size)
        concatenated_weights = self.aggregator._concatenate_weights([sliced_weights] * self.world_size)
        
        # Now ALL parameters should be handled, including num_batches_tracked
        original_keys = set(original_weights.keys())
        concat_keys = set(concatenated_weights.keys())
        slice_keys = set(sliced_weights.keys())
        
        missing_in_slice = original_keys - slice_keys
        missing_in_concat = original_keys - concat_keys
        extra_in_slice = slice_keys - original_keys
        extra_in_concat = concat_keys - original_keys
        
        errors = []
        if missing_in_slice:
            errors.append(f"Parameters missing in slice: {missing_in_slice}")
        if missing_in_concat:
            errors.append(f"Parameters missing in concatenation: {missing_in_concat}")
        if extra_in_slice:
            errors.append(f"Extra parameters in slice: {extra_in_slice}")
        if extra_in_concat:
            errors.append(f"Extra parameters in concatenation: {extra_in_concat}")
        
        # Count parameters that are handled
        total_params = len(original_keys)
        handled_params = len(concat_keys)
        
        if errors:
            print(f"❌ Parameter coverage test failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("✅ Parameter coverage test passed!")
            print(f"Total parameters: {total_params}")
            print(f"Parameters handled: {handled_params}")
            return True
    
    def test_tensor_shapes(self):
        """Test that tensor shapes are preserved correctly."""
        print("\nTesting tensor shapes...")
        
        # Get original weights
        original_weights = self.full_model.state_dict()
        
        # Slice weights for each trainer
        sliced_weights = []
        for rank in range(self.world_size):
            sliced = self.aggregator._slice_weights(original_weights, rank, self.world_size)
            sliced_weights.append(sliced)
        
        # Test shape consistency for each parameter type
        errors = []
        
        for name, original_tensor in original_weights.items():
            # if 'num_batches_tracked' in name:
            #     continue  # Skip num_batches_tracked parameters
                
            original_shape = original_tensor.shape
            
            if name == "conv1.weight":
                # Should be split by output channels
                expected_slice_shape = (64 // self.world_size, 3, 7, 7)
                for rank in range(self.world_size):
                    actual_shape = sliced_weights[rank][name].shape
                    if actual_shape != expected_slice_shape:
                        errors.append(f"{name} rank {rank}: expected {expected_slice_shape}, got {actual_shape}")
            
            elif 'num_batches_tracked' in name:
                expected_slice_shape = ()
                for rank in range(self.world_size):
                    actual_shape = sliced_weights[rank][name].shape
                    if actual_shape != expected_slice_shape:
                        errors.append(f"{name} rank {rank}: expected {expected_slice_shape}, got {actual_shape}")

            elif name.startswith("bn1."):
                # Initial BN parameters should be split
                expected_slice_shape = (64 // self.world_size,)
                for rank in range(self.world_size):
                    actual_shape = sliced_weights[rank][name].shape
                    if actual_shape != expected_slice_shape:
                        errors.append(f"{name} rank {rank}: expected {expected_slice_shape}, got {actual_shape}")
            
            elif '.conv1.weight' in name and 'layer' in name:
                # BasicBlock conv1: split by output channels
                layer_channels = self.aggregator._get_layer_channels(name)
                expected_slice_shape = (layer_channels // self.world_size, original_shape[1], 3, 3)
                for rank in range(self.world_size):
                    actual_shape = sliced_weights[rank][name].shape
                    if actual_shape != expected_slice_shape:
                        errors.append(f"{name} rank {rank}: expected {expected_slice_shape}, got {actual_shape}")
            
            elif '.conv2.weight' in name and 'layer' in name:
                # BasicBlock conv2: split by input channels
                layer_channels = self.aggregator._get_layer_channels(name)
                expected_slice_shape = (original_shape[0], layer_channels // self.world_size, 3, 3)
                for rank in range(self.world_size):
                    actual_shape = sliced_weights[rank][name].shape
                    if actual_shape != expected_slice_shape:
                        errors.append(f"{name} rank {rank}: expected {expected_slice_shape}, got {actual_shape}")
            
            elif any(pattern in name for pattern in ['.bn1.weight', '.bn1.bias', '.bn1.running_mean', '.bn1.running_var']) and 'layer' in name:
                # BasicBlock bn1: split parameters
                layer_channels = self.aggregator._get_layer_channels(name)
                expected_slice_shape = (layer_channels // self.world_size,)
                for rank in range(self.world_size):
                    actual_shape = sliced_weights[rank][name].shape
                    if actual_shape != expected_slice_shape:
                        errors.append(f"{name} rank {rank}: expected {expected_slice_shape}, got {actual_shape}")

            else:
                # Full parameters (bn2, shortcut, fc) should remain unchanged
                for rank in range(self.world_size):
                    if name in sliced_weights[rank]:
                        actual_shape = sliced_weights[rank][name].shape
                        if actual_shape != original_shape:
                            errors.append(f"{name} rank {rank}: expected {original_shape}, got {actual_shape}")
        
        if errors:
            print(f"❌ Shape test failed with {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
            return False
        else:
            print("✅ All tensor shapes are correct!")
            return True
    
    def test_round_trip_preservation(self):
        """Test that slice -> concatenate preserves original values for split parameters."""
        print("\nTesting round-trip preservation...")
        
        # Get original weights
        original_weights = self.full_model.state_dict()
        
        # Slice weights for each trainer
        sliced_weights = []
        for rank in range(self.world_size):
            sliced = self.aggregator._slice_weights(original_weights, rank, self.world_size)
            sliced_weights.append(sliced)
        
        # Concatenate weights back
        concatenated_weights = self.aggregator._concatenate_weights(sliced_weights)
        
        # Check if split parameters are preserved exactly
        errors = []
        split_params = []
        
        for name, original_tensor in original_weights.items():
            if name in concatenated_weights:
                concatenated_tensor = concatenated_weights[name]
                
                # Determine if this parameter should be preserved exactly or averaged
                is_split_param = (
                    name == "conv1.weight" or
                    name.startswith("bn1.") or
                    ('.conv1.weight' in name and 'layer' in name) or
                    ('.conv2.weight' in name and 'layer' in name) or
                    (any(pattern in name for pattern in ['.bn1.weight', '.bn1.bias', '.bn1.running_mean', '.bn1.running_var']) and 'layer' in name)
                )
                
                if is_split_param:
                    split_params.append(name)
                    # For split parameters, concatenation should recover original exactly
                    if not torch.allclose(original_tensor, concatenated_tensor, atol=1e-6):
                        max_diff = torch.max(torch.abs(original_tensor - concatenated_tensor)).item()
                        errors.append(f"{name}: max difference = {max_diff}")
                else:
                    # For averaged parameters, we expect them to be identical (since we start with same weights)
                    if not torch.allclose(original_tensor, concatenated_tensor, atol=1e-6):
                        max_diff = torch.max(torch.abs(original_tensor - concatenated_tensor)).item()
                        errors.append(f"{name} (averaged): max difference = {max_diff}")
        
        print(f"Split parameters tested: {len(split_params)}")
        print(f"Total parameters tested: {len(concatenated_weights)}")
        
        if errors:
            print(f"❌ Round-trip test failed with {len(errors)} errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
            return False
        else:
            print("✅ Round-trip preservation test passed!")
            return True
    
    def test_averaging_behavior(self):
        """Test that replicated parameters are averaged correctly."""
        print("\nTesting averaging behavior...")
        
        # Create different weights for each trainer to test averaging
        original_weights = self.full_model.state_dict()
        modified_weights = []
        
        for rank in range(self.world_size):
            weights_copy = deepcopy(original_weights)
            # Modify replicated parameters with different values
            for name, tensor in weights_copy.items():
                is_replicated = (
                    '.bn2' in name or 
                    '.shortcut' in name or 
                    'fc.' in name
                )
                # Skip num_batches_tracked as it's integer and doesn't make sense to add float offsets
                if is_replicated and 'num_batches_tracked' not in name:
                    # Add different offsets for each rank
                    weights_copy[name] = tensor + (rank + 1) * 0.1
            
            # Now slice these modified weights
            sliced = self.aggregator._slice_weights(weights_copy, rank, self.world_size)
            modified_weights.append(sliced)
        
        # Concatenate the modified weights
        concatenated_weights = self.aggregator._concatenate_weights(modified_weights)
        
        # Check averaging for replicated parameters
        errors = []
        
        for name, original_tensor in original_weights.items():
            if name in concatenated_weights:
                concatenated_tensor = concatenated_weights[name]
                
                is_replicated = (
                    '.bn2' in name or 
                    '.shortcut' in name or 
                    'fc.' in name
                )
                
                # Only test averaging for parameters we actually modified (exclude num_batches_tracked)
                if is_replicated and 'num_batches_tracked' not in name:
                    # Calculate expected average
                    expected_average = original_tensor + (1 + self.world_size) * 0.1 / 2
                    
                    if not torch.allclose(concatenated_tensor, expected_average, atol=1e-6):
                        max_diff = torch.max(torch.abs(concatenated_tensor - expected_average)).item()
                        errors.append(f"{name}: averaging error, max difference = {max_diff}")
        
        if errors:
            print(f"❌ Averaging test failed with {len(errors)} errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("✅ Averaging behavior test passed!")
            return True
    
    def test_different_world_sizes(self):
        """Test operations with different world sizes."""
        print("\nTesting different world sizes...")
        
        world_sizes = [2, 4, 8]  # Test common world sizes
        results = []
        
        for ws in world_sizes:
            try:
                # Create aggregator for this world size
                class DummyConfig:
                    class Hyperparameters:
                        seed = 42
                        world_size = ws
                        learning_rate = 0.01
                        rounds = 1
                        pretrain = False
                    hyperparameters = Hyperparameters()
                
                config = DummyConfig()
                aggregator = agg_main.Cifar100ResNet34Aggregator(config)
                aggregator.world_size = ws
                
                # Test basic functionality
                original_weights = self.full_model.state_dict()
                sliced_weights = []
                
                for rank in range(ws):
                    sliced = aggregator._slice_weights(original_weights, rank, ws)
                    sliced_weights.append(sliced)
                
                concatenated = aggregator._concatenate_weights(sliced_weights)
                
                # Check that we get expected parameters back (all parameters now)
                original_keys = set(original_weights.keys())
                concat_keys = set(concatenated.keys())
                
                if original_keys == concat_keys:
                    print(f"  ✅ World size {ws}: OK")
                    results.append(True)
                else:
                    missing = original_keys - concat_keys
                    extra = concat_keys - original_keys
                    print(f"  ❌ World size {ws}: missing {len(missing)}, extra {len(extra)}")
                    results.append(False)

                # Make sure tensor value of each layer match
                for name, original_tensor in original_weights.items():
                    concat_tensor = concatenated[name]

                    if not torch.allclose(original_tensor, concat_tensor, atol=1e-6):
                        max_diff = torch.max(torch.abs(original_tensor - concat_tensor)).item()
                        print(f"{name}: max difference = {max_diff}")
                        results.append(False)
                
            except Exception as e:
                print(f"  ❌ World size {ws}: {str(e)}")
                results.append(False)
        
        if all(results):
            print("✅ Different world sizes test passed!")
            return True
        else:
            print(f"❌ Different world sizes test failed")
            return False
    
    def test_concatenation_consistency(self):
        """Test that concatenation operations work consistently."""
        print("\nTesting concatenation consistency...")
        
        original_weights = self.full_model.state_dict()
        errors = []
        
        # Test that split and concatenated tensors have correct dimensions
        for name, tensor in original_weights.items():
            try:
                # Get sliced weights for all ranks
                sliced_list = []
                for rank in range(self.world_size):
                    sliced = self.aggregator._slice_weights(original_weights, rank, self.world_size)
                    if name in sliced:
                        sliced_list.append(sliced[name])
                
                if not sliced_list:
                    continue
                    
                # Test concatenation logic
                if (name == "conv1.weight" or 
                    (name.startswith("bn1.") and 'num_batches_tracked' not in name)):
                    # Should concatenate along dimension 0
                    try:
                        concatenated = torch.cat(sliced_list, 0)
                        if concatenated.shape != tensor.shape:
                            errors.append(f"{name}: concat shape mismatch {concatenated.shape} vs {tensor.shape}")
                    except Exception as e:
                        errors.append(f"{name}: concat error - {str(e)}")
                        
                elif '.conv1.weight' in name and 'layer' in name:
                    # Should concatenate along dimension 0 (output channels)
                    try:
                        concatenated = torch.cat(sliced_list, 0)
                        if concatenated.shape != tensor.shape:
                            errors.append(f"{name}: concat shape mismatch {concatenated.shape} vs {tensor.shape}")
                    except Exception as e:
                        errors.append(f"{name}: concat error - {str(e)}")
                        
                elif '.conv2.weight' in name and 'layer' in name:
                    # Should concatenate along dimension 1 (input channels)
                    try:
                        concatenated = torch.cat(sliced_list, 1)
                        if concatenated.shape != tensor.shape:
                            errors.append(f"{name}: concat shape mismatch {concatenated.shape} vs {tensor.shape}")
                    except Exception as e:
                        errors.append(f"{name}: concat error - {str(e)}")
                        
            except Exception as e:
                errors.append(f"{name}: general error - {str(e)}")
        
        if errors:
            print(f"❌ Concatenation consistency test failed with {len(errors)} errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
            return False
        else:
            print("✅ Concatenation consistency test passed!")
            return True
    
    def print_all_layer_shapes(self):
        """Print the shapes of all layers in the model."""
        print("\n" + "="*60)
        print("MODEL LAYER SHAPES")
        print("="*60)
        
        model_state = self.full_model.state_dict()
        
        # Group parameters by layer type for better organization
        conv_params = []
        bn_params = []
        fc_params = []
        other_params = []
        
        for name, tensor in model_state.items():
            if 'conv' in name:
                conv_params.append((name, tensor.shape))
            elif any(bn_type in name for bn_type in ['bn', 'batch_norm']):
                bn_params.append((name, tensor.shape))
            elif 'fc' in name or 'linear' in name:
                fc_params.append((name, tensor.shape))
            else:
                other_params.append((name, tensor.shape))
        
        # Print convolution layers
        if conv_params:
            print("\n🔷 Convolution Layers:")
            for name, shape in conv_params:
                print(f"  {name:<35} : {str(shape)}")
        
        # Print batch normalization layers
        if bn_params:
            print("\n🔶 Batch Normalization Layers:")
            for name, shape in bn_params:
                print(f"  {name:<35} : {str(shape)}")
        
        # Print fully connected layers
        if fc_params:
            print("\n🔸 Fully Connected Layers:")
            for name, shape in fc_params:
                print(f"  {name:<35} : {str(shape)}")
        
        # Print other parameters
        if other_params:
            print("\n🔹 Other Parameters:")
            for name, shape in other_params:
                print(f"  {name:<35} : {str(shape)}")
        
        # Print summary statistics
        total_params = sum(tensor.numel() for tensor in model_state.values())
        total_layers = len(model_state)
        
        print("\n" + "-"*60)
        print(f"📊 Summary:")
        print(f"  Total layers: {total_layers}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Model size (MB): {total_params * 4 / (1024**2):.2f}")  # Assuming float32
        print("="*60)
    
    def run_all_tests(self):
        """Run all tests and return overall result."""
        print("="*60)
        print("RUNNING FIXED WEIGHT OPERATIONS TESTS FOR RESNET34")
        print("="*60)
        
        set_seed(42)  # Ensure reproducibility
        
        tests = [
            ("Parameter Coverage", self.test_parameter_coverage),
            ("Tensor Shapes", self.test_tensor_shapes),
            ("Concatenation Consistency", self.test_concatenation_consistency),
            ("Round-trip Preservation", self.test_round_trip_preservation),
            ("Averaging Behavior", self.test_averaging_behavior),
            ("Different World Sizes", self.test_different_world_sizes),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:.<40} {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! The weight operations are working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Please check the implementation.")
            return False


def print_model_layer_shapes(model):
    """
    Standalone function to print the shapes of all layers in a PyTorch model.
    
    Args:
        model: PyTorch model to analyze
    """
    print("\n" + "="*60)
    print("MODEL LAYER SHAPES")
    print("="*60)
    
    model_state = model.state_dict()
    
    # Group parameters by layer type for better organization
    conv_params = []
    bn_params = []
    fc_params = []
    other_params = []
    
    for name, tensor in model_state.items():
        if 'conv' in name:
            conv_params.append((name, tensor.shape))
        elif any(bn_type in name for bn_type in ['bn', 'batch_norm']):
            bn_params.append((name, tensor.shape))
        elif 'fc' in name or 'linear' in name:
            fc_params.append((name, tensor.shape))
        else:
            other_params.append((name, tensor.shape))
    
    # Print convolution layers
    if conv_params:
        print("\n🔷 Convolution Layers:")
        for name, shape in conv_params:
            print(f"  {name:<35} : {str(shape)}")
    
    # Print batch normalization layers
    if bn_params:
        print("\n🔶 Batch Normalization Layers:")
        for name, shape in bn_params:
            print(f"  {name:<35} : {str(shape)}")
    
    # Print fully connected layers
    if fc_params:
        print("\n🔸 Fully Connected Layers:")
        for name, shape in fc_params:
            print(f"  {name:<35} : {str(shape)}")
    
    # Print other parameters
    if other_params:
        print("\n🔹 Other Parameters:")
        for name, shape in other_params:
            print(f"  {name:<35} : {str(shape)}")
    
    # Print summary statistics
    total_params = sum(tensor.numel() for tensor in model_state.values())
    total_layers = len(model_state)
    
    print("\n" + "-"*60)
    print(f"📊 Summary:")
    print(f"  Total layers: {total_layers}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size (MB): {total_params * 4 / (1024**2):.2f}")  # Assuming float32
    print("="*60)


def main():
    """Main function to run the tests."""
    # Test with world_size=2 by default
    tester = WeightOperationsTester(world_size=2)
    
    # Print layer shapes first
    print("🔍 Analyzing model architecture...")
    tester.print_all_layer_shapes()
    
    # Also demonstrate the standalone function
    print("\n🔍 Using standalone function:")
    print_model_layer_shapes(tester.full_model)
    
    # Run all tests
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
