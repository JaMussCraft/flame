"""Test file for validating _slice_weights and concatenation logic."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from copy import deepcopy


class Net(nn.Module):
    """Net class."""

    def __init__(self):
        """Initialize."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def _slice_weights(state_dict, rank, world_size):
    """Slice weights for a specific rank."""
    sliced = {}
    for name, full_tensor in state_dict.items():
        if name == "conv1.weight":
            slice_size = 32 // world_size
            sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size].clone()
        elif name == "conv1.bias":
            slice_size = 32 // world_size
            sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size].clone()
        elif name == "conv2.weight":
            out_slice_size = 64 // world_size
            in_slice_size = 32 // world_size
            # print(f"{rank * out_slice_size}:{(rank + 1) * out_slice_size}, {rank * in_slice_size}:{(rank + 1) * in_slice_size}")
            sliced[name] = full_tensor[rank * out_slice_size:(rank + 1) * out_slice_size,
                                    rank * in_slice_size:(rank + 1) * in_slice_size].clone()
        elif name == "conv2.bias":
            slice_size = 64 // world_size
            sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size].clone()
        elif name == "fc1.weight":
            # # OLD
            # out_slice_size = 128 // world_size
            # in_slice_size = (64 // world_size) * 12 * 12
            # sliced[name] = full_tensor[rank * out_slice_size:(rank + 1) * out_slice_size,
            #                         :in_slice_size].clone()
            
            # NEW
            row_chunks = torch.chunk(full_tensor, world_size//2, dim=0) 
            shards = [shard for chunk in row_chunks for shard in torch.chunk(chunk, world_size//2, dim=1)]
            print("SHARKS: ", shards[rank].shape)
            sliced[name] = shards[rank]



        elif name == "fc1.bias":
            slice_size = 128 // world_size
            sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size].clone()
        elif name == "fc2.weight":
            slice_size = 128 // world_size
            sliced[name] = full_tensor[:, rank * slice_size:(rank + 1) * slice_size].clone()
        elif name == "fc2.bias":
            sliced[name] = full_tensor.clone()  # Not split, same for all trainers
    return sliced


def test_concatenation_logic(appended_weights, world_size):
    """Test the concatenation logic from _aggregate_weights."""
    concated = {}
    
    start_time = time.time()
    
    # Concatenate conv1.weight
    weights = []
    for w in appended_weights:
        weights.append(w['conv1.weight'])
    concated['conv1.weight'] = torch.cat(weights, 0)
    
    # Concatenate conv1.bias
    weights = []
    for w in appended_weights:
        weights.append(w['conv1.bias'])
    concated['conv1.bias'] = torch.cat(weights, 0)

    # Concatenate conv2.weight
    weights_list = [w['conv2.weight'] for w in appended_weights] 

    # Old Implementation
    # concated['conv2.weight'] = torch.cat(weights_list, dim=0)  
    # if concated['conv2.weight'].shape[1] != 32:
    #     concated['conv2.weight'] = torch.cat([concated['conv2.weight']] * (32 // concated['conv2.weight'].shape[1]), dim=1)  
    
    # NEW Implementation
    out_slice_size = 64 // world_size
    in_slice_size = 32 // world_size        
    # Create empty tensor with original shape
    original_shape = (64, 32, 3, 3)  # or get from original weights
    reconstructed = torch.zeros(original_shape, dtype=weights_list[0].dtype)

    # Place each rank's slice back in its correct position
    for rank, weight_slice in enumerate(weights_list):
        out_start = rank * out_slice_size
        out_end = (rank + 1) * out_slice_size
        in_start = rank * in_slice_size
        in_end = (rank + 1) * in_slice_size
        
        reconstructed[out_start:out_end, in_start:in_end] = weight_slice.clone()

    concated['conv2.weight'] = reconstructed


    # Concatenate conv2.bias
    weights = []
    for w in appended_weights:
        weights.append(w['conv2.bias'])
    concated['conv2.bias'] = torch.cat(weights, 0)

    # Concatenate fc1.weight
    weights_list = [w['fc1.weight'] for w in appended_weights]  
    concated['fc1.weight'] = torch.cat(weights_list, dim=0)  
    if concated['fc1.weight'].shape[1] != 9216:
        concated['fc1.weight'] = torch.cat([concated['fc1.weight']] * (9216 // concated['fc1.weight'].shape[1]), dim=1)

    # Concatenate fc1.bias
    weights = []
    for w in appended_weights:
        weights.append(w['fc1.bias'])
    concated['fc1.bias'] = torch.cat(weights, 0)

    # Concatenate fc2.weight
    weights = []
    for w in appended_weights:
        weights.append(w['fc2.weight'])

    # OLD
    concated['fc2.weight'] = torch.cat(weights, 1)

    # NEW
    # reconstructed = torch.zeros((10, 128), dtype=weights[0].dtype)
    # slice_size = 128 // world_size
    # for rank in range(world_size):
    #     reconstructed[:, rank * slice_size:(rank + 1) * slice_size] = weights[rank]

    # concated['fc2.weight'] = reconstructed

    # Average fc2.bias
    weights_list = [w['fc2.bias'] for w in appended_weights] 
    concated['fc2.bias'] = torch.mean(torch.stack(weights_list), dim=0) 

    end_time = time.time()
    con_time = end_time - start_time
    
    return concated, con_time


def print_tensor_shapes(state_dict, title="Tensor Shapes"):
    """Print shapes of all tensors in state dict."""
    print(f"\n{title}:")
    print("-" * 50)
    for name, tensor in state_dict.items():
        print(f"{name}: {tensor.shape}")


def print_tensors_values(state_dict, title="Tensors Values", max_elements=20):
    """Print actual values of tensors (useful for integer debugging)."""
    print(f"\n{title}:")
    print("-" * 50)
    for name, tensor in state_dict.items():
        flat_tensor = tensor.flatten()
        if len(flat_tensor) <= max_elements:
            print(f"{name}: {flat_tensor.tolist()}")
        else:
            print(f"{name}: {flat_tensor[:max_elements].tolist()}... (showing first {max_elements} elements)")
        print(f"  Shape: {tensor.shape}, Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}")
        print()

def print_tensor_values(tensor, title="Tensor Values", max_elements=20):
    """Print actual values of tensors (useful for integer debugging)."""
    print(f"\n{title}:")
    print("-" * 50)

    flat_tensor = tensor.flatten()
    if len(flat_tensor) <= max_elements:
        print(f"Tensor values: {flat_tensor.tolist()}")
    else:
        print(f"Tensor values: {flat_tensor[:max_elements].tolist()}... (showing first {max_elements} elements)")
    print(f"  Shape: {tensor.shape}, Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}")
    print(f"  Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")
    print()


def compare_tensors(original_weights, reconstructed_weights, tolerance=1e-6):
    """Compare original and reconstructed tensors for exact matches."""
    print("\nComparing tensor values:")
    print("-" * 50)
    all_match = True
    
    for name in original_weights.keys():
        orig_tensor = original_weights[name]
        recon_tensor = reconstructed_weights[name]
        
        # Check if tensors are close within tolerance
        if torch.allclose(orig_tensor, recon_tensor, atol=tolerance):
            print(f"{name}: ✓ EXACT MATCH")
        else:
            all_match = False
            max_diff = torch.max(torch.abs(orig_tensor - recon_tensor)).item()
            print(f"{name}: ✗ MISMATCH (max diff: {max_diff:.8f})")
    
    return all_match


def compare_tensors_detailed(tensor1, tensor2, tolerance=1e-6, max_mismatches=10, tensor1_name="Tensor1", tensor2_name="Tensor2"):
    """
    Compare two tensors and print exact locations of mismatched values.
    
    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        tolerance: Tolerance for floating point comparison
        max_mismatches: Maximum number of mismatches to print (to avoid spam)
        tensor1_name: Name for first tensor (for clearer output)
        tensor2_name: Name for second tensor (for clearer output)
    
    Returns:
        bool: True if tensors match within tolerance, False otherwise
    """
    print(f"\nDetailed comparison between {tensor1_name} and {tensor2_name}:")
    print("-" * 80)
    
    # Check if shapes match
    if tensor1.shape != tensor2.shape:
        print(f"❌ SHAPE MISMATCH:")
        print(f"  {tensor1_name} shape: {tensor1.shape}")
        print(f"  {tensor2_name} shape: {tensor2.shape}")
        return False
    
    print(f"✓ Shapes match: {tensor1.shape}")
    
    # Check if tensors are close within tolerance
    if torch.allclose(tensor1, tensor2, atol=tolerance):
        print(f"✅ TENSORS MATCH (within tolerance {tolerance})")
        return True
    
    # Find mismatched locations
    diff = torch.abs(tensor1 - tensor2)
    mismatch_mask = diff > tolerance
    
    # Get indices of mismatched elements
    mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)
    num_mismatches = len(mismatch_indices)
    
    print(f"❌ TENSORS DO NOT MATCH")
    print(f"Total mismatched elements: {num_mismatches} out of {tensor1.numel()}")
    print(f"Percentage mismatched: {100 * num_mismatches / tensor1.numel():.2f}%")
    
    # Statistics about differences
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    std_diff = torch.std(diff).item()
    
    print(f"\nDifference statistics:")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")
    print(f"  Std difference: {std_diff:.8f}")
    
    # Print detailed mismatch locations
    print(f"\nFirst {min(max_mismatches, num_mismatches)} mismatch locations:")
    print("Index".ljust(20), f"{tensor1_name} Value".ljust(15), f"{tensor2_name} Value".ljust(15), "Difference")
    print("-" * 70)
    
    for i in range(min(max_mismatches, num_mismatches)):
        idx = mismatch_indices[i]
        idx_tuple = tuple(idx.tolist())
        
        val1 = tensor1[idx_tuple].item()
        val2 = tensor2[idx_tuple].item()
        diff_val = abs(val1 - val2)
        
        # Format index nicely
        if len(idx_tuple) == 1:
            idx_str = f"[{idx_tuple[0]}]"
        else:
            idx_str = str(idx_tuple)
        
        print(f"{idx_str:<20} {val1:<15.6f} {val2:<15.6f} {diff_val:.6f}")
    
    if num_mismatches > max_mismatches:
        print(f"... and {num_mismatches - max_mismatches} more mismatches")
    
    return False


def compare_state_dicts_detailed(state_dict1, state_dict2, tolerance=1e-6, max_mismatches_per_layer=5):
    """
    Compare two state dictionaries and show detailed mismatches for each layer.
    
    Args:
        state_dict1: First state dictionary
        state_dict2: Second state dictionary
        tolerance: Tolerance for floating point comparison
        max_mismatches_per_layer: Maximum mismatches to show per layer
    
    Returns:
        bool: True if all tensors match, False otherwise
    """
    print("\nDetailed State Dictionary Comparison:")
    print("=" * 80)
    
    all_match = True
    
    # Check if keys match
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    if keys1 != keys2:
        print("❌ KEY MISMATCH:")
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            print(f"  Only in first dict: {only_in_1}")
        if only_in_2:
            print(f"  Only in second dict: {only_in_2}")
        return False
    
    # Compare each tensor
    for key in sorted(state_dict1.keys()):
        print(f"\n{'='*20} {key} {'='*20}")
        tensor_match = compare_tensors_detailed(
            state_dict1[key], 
            state_dict2[key], 
            tolerance=tolerance,
            max_mismatches=max_mismatches_per_layer,
            tensor1_name="Original",
            tensor2_name="Reconstructed"
        )
        all_match = all_match and tensor_match
    
    print(f"\n{'='*80}")
    print(f"Overall comparison result: {'✅ ALL MATCH' if all_match else '❌ MISMATCHES FOUND'}")
    
    return all_match

def run_test(use_integers=False):
    """Run the test to validate slicing and concatenation."""
    print("Testing _slice_weights and concatenation logic")
    print("=" * 60)
    
    # Initialize model with random weights
    model = Net()
    # Force initialization with specific seed for reproducibility
    torch.manual_seed(42)
    
    if use_integers:
        # Initialize with integer values for cleaner testing
        for param in model.parameters():
            param.data = torch.randint(-10, 11, param.shape, dtype=torch.float32)
        print_tensor_shapes(model.state_dict(), "Original Model Weights (Random Integer Initialization)")
    else:
        # Initialize with normal distribution
        for param in model.parameters():
            param.data.normal_(0, 0.1)
        print_tensor_shapes(model.state_dict(), "Original Model Weights (Random Float Initialization)")
    
    original_weights = model.state_dict()

    print_tensors_values(original_weights, title="Original Weights", max_elements=20)
    
    # Test with world_size = 4
    world_size = 4
    print(f"\nTesting with world_size = {world_size}")
    
    # Create sliced weights for each rank
    sliced_weights = []
    for rank in range(world_size):
        sliced = _slice_weights(original_weights, rank, world_size)
        sliced_weights.append(sliced)
        print_tensor_shapes(sliced, f"Sliced Weights for Rank {rank}")
    
    # Test concatenation
    print("\nTesting concatenation...")
    concatenated_weights, con_time = test_concatenation_logic(sliced_weights, world_size)
    
    print_tensor_shapes(concatenated_weights, "Concatenated Weights")
    print_tensors_values(original_weights, title="Concatenated Weights", max_elements=20)
    
    # Verify shapes match original
    print("\nVerifying shapes match original:")
    print("-" * 50)
    shapes_match = True
    for name in original_weights.keys():
        orig_shape = original_weights[name].shape
        concat_shape = concatenated_weights[name].shape
        match = orig_shape == concat_shape
        shapes_match = shapes_match and match
        print(f"{name}: Original {orig_shape} -> Concatenated {concat_shape} {'✓' if match else '✗'}")
    
    print(f"\nShape verification: {'PASSED' if shapes_match else 'FAILED'}")
    
    # Compare exact tensor values
    values_match = compare_tensors(original_weights, concatenated_weights)
    print(f"Value verification: {'PASSED' if values_match else 'FAILED'}")
    
    # If values don't match, show detailed comparison
    if not values_match:
        print("\n" + "="*60)
        print("DETAILED MISMATCH ANALYSIS:")
        compare_state_dicts_detailed(original_weights, concatenated_weights, tolerance=1e-6, max_mismatches_per_layer=3)
    
    print(f"\nOverall test result: {'PASSED' if (shapes_match and values_match) else 'FAILED'}")
    
    # Test with different world_size values
    print("\n" + "=" * 60)
    print("Testing with different world_size values:")
    
    for ws in [2, 8]:
        print(f"\nTesting world_size = {ws}")
        try:
            test_sliced = []
            for rank in range(ws):
                sliced = _slice_weights(original_weights, rank, ws)
                test_sliced.append(sliced)
            
            concat_weights, _ = test_concatenation_logic(test_sliced, ws)
            
            # Check if shapes match
            shapes_match = all(original_weights[name].shape == concat_weights[name].shape 
                             for name in original_weights.keys())
            
            # Check if values match
            values_match = all(torch.allclose(original_weights[name], concat_weights[name], atol=1e-6)
                             for name in original_weights.keys())
            
            overall_match = shapes_match and values_match
            print(f"World size {ws}: {'PASSED' if overall_match else 'FAILED'}")
            if not overall_match:
                print(f"  Shapes match: {shapes_match}")
                print(f"  Values match: {values_match}")
            
        except Exception as e:
            print(f"World size {ws}: ERROR - {e}")


def test_individual_layers(use_integers=False):
    """Test individual layer slicing and concatenation to identify issues."""
    print("\n" + "=" * 60)
    print("Testing individual layers:")
    
    model = Net()
    torch.manual_seed(42)
    
    if use_integers:
        # Initialize with integer values for cleaner testing
        for param in model.parameters():
            param.data = torch.randint(-10, 11, param.shape, dtype=torch.float32)
        print("Using integer initialization for cleaner debugging")
    else:
        # Initialize with normal distribution
        for param in model.parameters():
            param.data.normal_(0, 0.1)
        print("Using float initialization")
    
    original_weights = model.state_dict()
    world_size = 4

    
    for layer_name in original_weights.keys():
        print(f"\nTesting {layer_name}:")
        print(f"  Original shape: {original_weights[layer_name].shape}")


        
        # Create sliced weights for this layer only
        sliced_tensors = []
        for rank in range(world_size):
            full_sliced = _slice_weights(original_weights, rank, world_size)
            sliced_tensors.append(full_sliced[layer_name])
            print(f"  Rank {rank} slice shape: {full_sliced[layer_name].shape}")
        
        # Test concatenation for this specific layer
        try:
            if layer_name in ["conv1.weight", "conv1.bias", "conv2.bias", "fc1.bias"]:
                # These are concatenated along dimension 0
                reconstructed = torch.cat(sliced_tensors, 0)
            elif layer_name == "conv2.weight":
                # OLD Implementation
                # Special handling for conv2.weight
                # reconstructed = torch.cat(sliced_tensors, dim=0)
                # if reconstructed.shape[1] != original_weights[layer_name].shape[1]:
                #     reconstructed = torch.cat([reconstructed] * (original_weights[layer_name].shape[1] // reconstructed.shape[1]), dim=1)

                # NEW Implementation
                out_slice_size = 64 // world_size
                in_slice_size = 32 // world_size        
                # Create empty tensor with original shape
                original_shape = (64, 32, 3, 3)  # or get from original weights
                reconstructed = torch.zeros(original_shape, dtype=sliced_tensors[0].dtype)

                # Place each rank's slice back in its correct position
                for rank, weight_slice in enumerate(sliced_tensors):
                    out_start = rank * out_slice_size
                    out_end = (rank + 1) * out_slice_size
                    in_start = rank * in_slice_size
                    in_end = (rank + 1) * in_slice_size
                    
                    reconstructed[out_start:out_end, in_start:in_end] = weight_slice.clone()

                print_tensor_values(reconstructed)
                print(reconstructed.mean().item(), reconstructed.std().item())

                # Compare conv2.weight weight_slice's individually   

                # Place each rank's slice back in its correct position
                for rank in range(world_size):
                    out_start = rank * out_slice_size
                    out_end = (rank + 1) * out_slice_size
                    in_start = rank * in_slice_size
                    in_end = (rank + 1) * in_slice_size
                    
                    original_slice = original_weights[layer_name][out_start:out_end, in_start:in_end]
                    contcat_slice = reconstructed[out_start:out_end, in_start:in_end]
                    
                    values_match = torch.allclose(original_slice, contcat_slice, atol=0)
                    print(f"TWO SLICES' VALUES MATCH: {'PASSED' if values_match else 'FAILED'}")

            elif layer_name == "fc1.weight":
                # Special handling for fc1.weight
                reconstructed = torch.cat(sliced_tensors, dim=0)
                if reconstructed.shape[1] != original_weights[layer_name].shape[1]:
                    reconstructed = torch.cat([reconstructed] * (original_weights[layer_name].shape[1] // reconstructed.shape[1]), dim=1)
            elif layer_name == "fc2.weight":
                # Concatenated along dimension 1
                reconstructed = torch.cat(sliced_tensors, 1)
            elif layer_name == "fc2.bias":
                # Averaged
                reconstructed = torch.mean(torch.stack(sliced_tensors), dim=0)
            
            print(f"  Reconstructed shape: {reconstructed.shape}")
            
            # Check if reconstruction matches original
            if reconstructed.shape == original_weights[layer_name].shape:
                if layer_name == "conv2.weight":
                    print_tensor_values(reconstructed, "YIPPIE")
                    print_tensor_values(original_weights[layer_name], "LESGO")
                    print(torch.allclose(original_weights[layer_name], reconstructed, atol=1))


                if torch.allclose(original_weights[layer_name], reconstructed, atol=1e-6):
                    print(f"  ✓ PERFECT RECONSTRUCTION")
                else:
                    max_diff = torch.max(torch.abs(original_weights[layer_name] - reconstructed)).item()
                    print(f"  ✗ VALUE MISMATCH (max diff: {max_diff:.8f})")
                    
                    # Show detailed comparison for this layer
                    print(f"\n  Detailed analysis for {layer_name}:")
                    compare_tensors_detailed(
                        original_weights[layer_name], 
                        reconstructed, 
                        tolerance=1e-6,
                        max_mismatches=5,
                        tensor1_name="Original",
                        tensor2_name="Reconstructed"
                    )
            else:
                print(f"  ✗ SHAPE MISMATCH")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test weight slicing and concatenation')
    parser.add_argument('--integers', action='store_true', 
                       help='Use integer initialization for cleaner debugging')
    args = parser.parse_args()
    
    # Run main test
    run_test(use_integers=args.integers)
    
    # Run individual layer test
    test_individual_layers(use_integers=args.integers)


