# Training Commands for AlexNet on ImageNet

## Prerequisites
1. Download ImageNet dataset from [imagenet](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and organize it as:
   ```
   /path/to/imagenet/
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

## Recommended Training Commands

### 1. Best Overall Performance (Recommended)
```bash
python train_imagenet.py --data-path /path/to/imagenet --num-classes 100 --batch-size 128 --epochs 100 --lr 0.01 --optimizer sgd --scheduler step --output-dir ./checkpoints_best
```
- **Expected**: ~70-80% top-1 accuracy on 100 classes
- **Training time**: ~6-8 hours on modern GPU

### 2. Faster Convergence
```bash
python train_imagenet.py --data-path /path/to/imagenet --num-classes 100 --batch-size 256 --epochs 80 --lr 0.02 --optimizer sgd --scheduler cosine --output-dir ./checkpoints_fast
```
- **Expected**: ~68-75% top-1 accuracy
- **Training time**: ~4-5 hours

### 3. More Stable Training
```bash
python train_imagenet.py --data-path /path/to/imagenet --num-classes 100 --batch-size 128 --epochs 120 --lr 0.005 --optimizer adamw --scheduler step --output-dir ./checkpoints_stable
```
- **Expected**: ~65-73% top-1 accuracy
- **Training time**: ~7-9 hours

### 4. Quick Test Run (50 classes)
```bash
python train_imagenet.py --data-path /path/to/imagenet --num-classes 50 --batch-size 128 --epochs 60 --lr 0.01 --optimizer sgd --scheduler step --output-dir ./checkpoints_test
```
- **Expected**: ~80-85% top-1 accuracy on 50 classes
- **Training time**: ~2-3 hours

### 5. High Performance (if you have good GPU memory)
```bash
python train_imagenet.py --data-path /path/to/imagenet --num-classes 100 --batch-size 256 --epochs 100 --lr 0.02 --optimizer sgd --scheduler step --output-dir ./checkpoints_high
```
- **Expected**: ~72-82% top-1 accuracy
- **Training time**: ~5-7 hours

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
    "log_interval": 100,
    "save_interval": 10,
    "output_dir": "./checkpoints",
    "seed": 42
}
```

Then run:
```bash
python train_imagenet.py --config config.json
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
