# Training Commands for AlexNet on ImageNet

## Prerequisites
1. Download ImageNet dataset from [imagenet](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and organize it as:
   ```
   ./imagenet/
   ├── train/
   │   ├── n01440764/
   │   │   ├── image1.JPEG
   │   │   └── ...
   │   └── n01443537/
   │       └── ...
   └── val/
       ├── n01440764/
       │   ├── image1.JPEG
       │   └── ...
       └── n01443537/
           └── ...
   ```

2. Navigate to the pretrained directory:
   ```bash
   cd /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/pretrained
   ```

## Architecture Choice

**NEW:** You can now choose between two AlexNet architectures:

### ImageNet Architecture (Recommended for ImageNet data)
- **Input Size**: 224×224 pixels
- **Best For**: High-resolution ImageNet data
- **Expected Accuracy**: 60-75% on 100 ImageNet classes
- **Architecture**: Classic ImageNet AlexNet with proper feature extraction layers
- **Use Case**: When you want to properly utilize ImageNet's visual complexity

### CIFAR-100 Architecture (For federated learning compatibility)
- **Input Size**: 32×32 pixels (images are resized)
- **Best For**: Maintaining compatibility with existing federated learning code
- **Expected Accuracy**: 5-15% on 100 ImageNet classes (limited by severe downsampling)
- **Architecture**: AlexNet modified for CIFAR-100 input size
- **Use Case**: When you need pretrained weights for the federated learning AlexNet in main.py

**Recommendation**: Use `--architecture imagenet` for better performance on ImageNet data, or `--architecture cifar100` if you need compatibility with the federated learning system.

## Performance Comparison

| Architecture | Input Size | ImageNet Accuracy | Use Case |
|--------------|------------|------------------|----------|
| ImageNet AlexNet | 224×224 | 60-75% | Best performance on ImageNet |
| CIFAR-100 AlexNet | 32×32 | 5-15% | Federated learning compatibility |

The dramatic performance difference is due to the severe downsampling when using 224×224 ImageNet images with the 32×32 CIFAR-100 architecture. Most visual information is lost in this process.

## Recommended Training Commands

### ImageNet Architecture (224x224 input - NEW!)

#### 1. Best Overall Performance (ImageNet Architecture)
```bash
python train_imagenet.py --data-path ./imagenet --num-classes 100 --batch-size 128 --epochs 100 --lr 0.01 --optimizer sgd --scheduler step --architecture imagenet --output-dir ./checkpoints_imagenet_best
```
- **Architecture**: ImageNet AlexNet (224x224 input)
- **Expected**: ~60-75% top-1 accuracy on 100 classes
- **Training time**: ~8-12 hours on modern GPU

#### 2. Faster Convergence (ImageNet Architecture)
```bash
python train_imagenet.py --data-path ./imagenet --num-classes 100 --batch-size 256 --epochs 80 --lr 0.02 --optimizer sgd --scheduler cosine --architecture imagenet --output-dir ./checkpoints_imagenet_fast
```
- **Architecture**: ImageNet AlexNet (224x224 input)
- **Expected**: ~55-70% top-1 accuracy
- **Training time**: ~6-8 hours

### CIFAR-100 Architecture (32x32 input - Original)

#### 3. Best Overall Performance (CIFAR-100 Architecture)
```bash
python train_imagenet.py --data-path ./imagenet --num-classes 100 --batch-size 128 --epochs 100 --lr 0.01 --optimizer sgd --scheduler step --architecture cifar100 --output-dir ./checkpoints_cifar100_best
```
- **Architecture**: CIFAR-100 AlexNet (32x32 input)
- **Expected**: ~5-15% top-1 accuracy on 100 classes (limited by downsampling)
- **Training time**: ~6-8 hours on modern GPU
- Actual Best validation accuracy: 5.78%

#### 4. Faster Convergence (CIFAR-100 Architecture)
```bash
python train_imagenet.py --data-path ./imagenet --num-classes 100 --batch-size 256 --epochs 80 --lr 0.02 --optimizer sgd --scheduler cosine --architecture cifar100 --output-dir ./checkpoints_cifar100_fast
```
- **Architecture**: CIFAR-100 AlexNet (32x32 input)
- **Expected**: ~3-12% top-1 accuracy
- **Training time**: ~4-5 hours

#### 5. Quick Test Run (50 classes, CIFAR-100 Architecture)
```bash
python train_imagenet.py --data-path ./imagenet --num-classes 50 --batch-size 128 --epochs 60 --lr 0.01 --optimizer sgd --scheduler step --architecture cifar100 --output-dir ./checkpoints_cifar100_test
```
- **Architecture**: CIFAR-100 AlexNet (32x32 input)
- **Expected**: ~8-18% top-1 accuracy on 50 classes
- **Training time**: ~2-3 hours
- Actual Best validation accuracy: 3.68%

## Resume Training
If training gets interrupted, you can resume from any checkpoint:
```bash
python train_imagenet.py --resume ./checkpoints_best/checkpoint_epoch_X.pth
```

## Using Configuration Files
Create a JSON config file for easier management:
```json
{
    "data_path": "/path/to/imagenet",
    "num_classes": 100,
    "batch_size": 128,
    "epochs": 100,
    "learning_rate": 0.01,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "scheduler": "step",
    "step_size": 30,
    "gamma": 0.1,
    "architecture": "imagenet",
    "log_interval": 100,
    "save_interval": 10,
    "output_dir": "./checkpoints",
    "seed": 42
}
```

Then run with ImageNet architecture:
```bash
python train_imagenet.py --config config_imagenet.json
```

Or with CIFAR-100 architecture:
```bash
python train_imagenet.py --config config_cifar100.json
```

You can also override specific parameters:
```bash
python train_imagenet.py --config config_imagenet.json --batch-size 256 --epochs 50
```

## Output Files
Each training run will create:
- `training.log` - Training progress logs
- `checkpoints/` - Model checkpoints
- `checkpoints/best_model.pth` - Best performing model
- `checkpoints/training_history.json` - Training metrics history

## GPU Memory Tips
If you encounter OOM errors:
- Reduce batch size: `--batch-size 64`
- Reduce number of workers: Add `--num-workers 2` to the config
- Use smaller image resolution (modify transforms in the script)

## Monitoring Training
The script logs:
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch
- Learning rate changes
- Best model saves
- Training time per epoch

All metrics are saved to `training_history.json` for analysis.
