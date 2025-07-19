# CIFAR100 AlexNet Experiment Automation

This directory contains scripts to automate CIFAR100 AlexNet experiments with different configurations and analyze the results.

## Files

1. **`run_experiments.py`** - Main automation script that runs experiments with different hyperparameter combinations
2. **`analyze_results.py`** - Script to analyze and visualize experiment results
3. **`experiment_results.pkl`** - Pickle file containing all experiment results (generated after running experiments)

## Usage

### Running Experiments

To run all experiment combinations:

```bash
python run_experiments.py
```

The script will automatically:
- Generate different combinations of hyperparameters
- Run experiments as background processes
- Save results in pickle format for easy analysis
- Handle process management and cleanup

### Experiment Configuration

The script tests the following hyperparameter combinations:

- **World sizes**: [1, 2] (number of trainers)
- **Learning rates**: [0.00001, 0.0001, 0.001]
- **Enable swapping**: [False, True]
- **Rounds**: [30, 50]
- **Seeds**: [123, 456, 789] (for reproducibility)

You can modify these lists in the `ExperimentRunner` class constructor in `run_experiments.py`.

### Analyzing Results

After running experiments, analyze the results with:

```bash
python analyze_results.py

python analyze_results.py --results-file ./aggregator/experiment_results.pkl --no-plots
```

This will generate:
- Summary statistics for all experiments
- Learning curves showing accuracy over rounds
- Hyperparameter comparison plots
- Best configuration rankings

For command-line options:
```bash
python analyze_results.py --help
```

## Experiment Process

For each experiment configuration, the script:

1. **Starts metaserver**: `sudo /home/cc/.flame/bin/metaserver`
2. **Starts aggregator**: `python aggregator/main.py <config.json>`
3. **Starts trainer(s)**: `python trainer/main.py <config.json>` (one per world_size)

All processes run in the background and are automatically cleaned up after completion.

## Results Format

The modified aggregator automatically saves results in a pickle file with the following structure:

```python
{
    (world_size, learning_rate, enable_swapping, rounds, seed): [
        (round_0, test_loss_0, test_accuracy_0),
        (round_1, test_loss_1, test_accuracy_1),
        ...
        (round_n, test_loss_n, test_accuracy_n)
    ]
}
```

The automation script simply tracks which experiments have been completed, while the actual results are saved directly by the aggregator.

## Configuration Files

The script automatically generates temporary configuration files for each experiment:
- `aggregator/config_temp.json` - Aggregator configuration
- `trainer/config_temp_{id}.json` - Trainer configurations

These are cleaned up after each experiment.

## Notes

- The script skips swapping experiments when world_size=1 (not applicable)
- Existing experiment results are preserved and not re-run
- All processes run with proper cleanup to avoid zombie processes
- The aggregator automatically saves results to `experiment_results.pkl`
- The automation script only tracks experiment completion status
- Text files are still generated for compatibility but analysis uses the pickle file

## Dependencies

Required Python packages:
- torch
- torchvision
- numpy
- pandas (for analysis)
- matplotlib (for analysis)
- seaborn (for analysis)
- pickle (built-in)

## Troubleshooting

- Make sure the metaserver binary exists at `/home/cc/.flame/bin/metaserver`
- Ensure you have sudo privileges to run the metaserver
- Check that all Python dependencies are installed
- If experiments hang, check that ports are not already in use

## Example Output

After running experiments, you'll see output like:

```
Total experiments to run: 45
=== Running experiment 1/45 ===
Starting experiment: world_size=1, lr=1e-05, swap=False, rounds=30, seed=123
Starting metaserver...
Starting aggregator...
Starting trainer 1 (rank 0)...
Waiting for training to complete...
Experiment completed successfully: (1, 1e-05, False, 30, 123)
```

The analysis script will then show:

```
Top 5 Configurations by Final Accuracy:
1. World Size: 2, LR: 0.001, Swap: False, Rounds: 50, Accuracy: 0.4523
2. World Size: 1, LR: 0.001, Swap: False, Rounds: 50, Accuracy: 0.4234
...
```
