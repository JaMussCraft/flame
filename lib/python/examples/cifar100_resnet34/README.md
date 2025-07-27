# CIFAR-100 ResNet34 with Tensor Parallelism

This example demonstrates federated learning with ResNet34 on CIFAR-100 using horizontal tensor parallelism. The implementation splits BasicBlocks across multiple trainers while maintaining compatibility with ImageNet pretrained weights.

## Architecture

### ResNet34 Structure
- **Initial Layer**: 7x7 conv (full channels) + BN + MaxPool
- **Layer1**: 3 BasicBlocks (64 channels)
- **Layer2**: 4 BasicBlocks (128 channels)
- **Layer3**: 6 BasicBlocks (256 channels)
- **Layer4**: 3 BasicBlocks (512 channels)
- **Final Layer**: AdaptiveAvgPool + FC (512→100 classes, not split)

### Tensor Parallelism Strategy

**Initial Layers (conv1, bn1):**
- Full parameters replicated on all trainers (not split)
- Averaged during aggregation for improved pretrained weight compatibility

Each BasicBlock follows this split pattern:
```
Input (full) → conv1 (split output) → bn1 (split params) → ReLU 
           → conv2 (split input → full output) → bn2 (full params) 
           + shortcut (full, replicated) → ReLU → Output (full)
```

**Key Design Decisions:**
- **Initial conv1/bn1**: Full parameters (replicated, averaged during aggregation)
- **BasicBlock conv1**: Split output channels across trainers
- **BasicBlock conv2**: Split input channels, produce full output for residual addition
- **BasicBlock bn1**: Split parameters (follows conv1 splits)
- **BasicBlock bn2**: Replicate parameters (follows conv2 full output)
- **Shortcut**: Full computation on all trainers (averaged during aggregation)

### Weight Distribution Strategy

**Slicing (Aggregator → Trainers):**
- Split layers: Slice tensors by channel dimension (BasicBlock conv1/bn1, conv2 input)
- Full layers: Send complete tensors to all trainers (initial conv1/bn1, FC layer, shortcut, bn2)

**Concatenation (Trainers → Aggregator):**
- Split layers: Concatenate tensors along split dimension (BasicBlock conv1/bn1, conv2 input)
- Full layers: Average tensors across trainers (initial conv1/bn1, FC layer, shortcut, bn2)

## Features

### Pretrained Weight Loading
- **Aggregator** loads ImageNet pretrained ResNet34 weights from torchvision
- Initializes final FC layer randomly for CIFAR-100 (100 classes)
- **FLAME framework** handles splitting and distributing weights to trainers

### Data Preprocessing
- Resizes CIFAR-100 images from 32x32 to 224x224 (ImageNet size)
- Uses ImageNet normalization statistics
- Applies data augmentation (random crop, horizontal flip)

### Parameter Splitting Coverage
- **~95% of parameters are split** across trainers
- Efficient memory and computation distribution
- All trainers remain active throughout training

## Configuration

### Hyperparameters
- `world_size`: Number of trainers (default: 2)
- `epochs`: Training epochs per round (default: 10)
- `rounds`: Total federated rounds (default: 10)
- `batch_size`: Batch size per trainer (default: 32)
- `learning_rate`: Learning rate (default: 0.01)
- `pretrained`: Load ImageNet weights (default: true)
- `seed`: Random seed for reproducibility (default: 42)

### Requirements
- PyTorch with torchvision
- FLAME federated learning framework
- CUDA-capable GPU (recommended)

## Usage

1. **Start Aggregator:**
   ```bash
   cd aggregator/
   python main.py config.json
   ```

2. **Start Trainers:** (in separate terminals)
   ```bash
   cd trainer/
   # Trainer 0
   python main.py config.json
   
   # Trainer 1 (modify rank in config)
   python main.py config_rank1.json
   ```

## Results

The implementation tracks:
- Test accuracy and loss per round
- Weight concatenation time
- Training convergence metrics

Results are saved to:
- `eval_res_ws{world_size}_r{rounds}.txt`: Detailed evaluation logs
- `experiment_results.pkl`: Experiment data for analysis

## Technical Notes

### Batch Normalization Handling
- **Split BN**: Parameters split along channel dimension (bn1)
- **Full BN**: Parameters replicated across trainers (bn2)
- **Aggregation**: Split params concatenated, full params averaged

### Memory Efficiency
- Each trainer holds ~1/world_size of total parameters
- Activation memory scales with split factor
- Communication overhead minimized through structured splitting

### Numerical Stability
- All computations maintain full precision
- Residual connections work correctly with alternating splits
- Batch statistics properly maintained across splits
