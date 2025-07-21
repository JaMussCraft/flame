#!/usr/bin/env python3
"""
Analysis script for CIFAR100 AlexNet experiment results.

This script loads and analyzes the experiment results saved in pickle format.

NEW FUNCTIONALITY:
- plot_specific_experiments(): Plot learning curves for experiments matching specific criteria
- Interactive plotting mode: Use --plot-specific flag for guided plotting
- Convenience functions: plot_by_world_size(), plot_by_learning_rate(), plot_swapping_comparison()

USAGE EXAMPLES:
1. Interactive plotting:
   python analyze_results.py --plot-specific

2. Programmatic plotting:
   from analyze_results import plot_specific_experiments, load_experiment_results, results_to_dataframe
   
   results = load_experiment_results("experiment_results.pkl")
   df = results_to_dataframe(results)
   
   # Plot specific configurations
   plot_specific_experiments(df, 
                           world_sizes=[2, 4], 
                           learning_rates=[0.0001, 0.001],
                           enable_swapping=[True, False])

3. Using convenience functions:
   from analyze_results import plot_by_world_size, plot_swapping_comparison
   
   plot_by_world_size([2, 4, 8])
   plot_swapping_comparison(world_sizes=[2, 4])
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Any
import os

def load_experiment_results(results_file: str = "experiment_results.pkl") -> Dict:
    """Load experiment results from pickle file."""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return {}
    
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)} experiments from {results_file}")
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return {}

def results_to_dataframe(results_dict: Dict) -> pd.DataFrame:
    """Convert results dictionary to pandas DataFrame for analysis."""
    rows = []
    
    for key, experiment_results in results_dict.items():
        # world_size, learning_rate, enable_swapping, enable_layerwise_swapping, rounds, seed = key
        world_size, learning_rate, enable_swapping, rounds, seed = key
        
        for round_num, test_loss, test_accuracy in experiment_results:
            rows.append({
                'world_size': world_size,
                'learning_rate': learning_rate,
                'enable_swapping': enable_swapping,
                'rounds': rounds,
                'seed': seed,
                'round': round_num,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            })
    
    return pd.DataFrame(rows)

def analyze_final_results(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze final results for each experiment configuration."""
    # Get the final round for each experiment
    final_results = df.groupby(['world_size', 'learning_rate', 'enable_swapping', 'rounds', 'seed']).last()
    
    # Calculate statistics
    stats = final_results.groupby(['world_size', 'learning_rate', 'enable_swapping', 'rounds', 'seed']).agg({
        'test_accuracy': ['mean', 'std', 'min', 'max'],
        'test_loss': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    return stats

def plot_learning_curves(df: pd.DataFrame, save_plots: bool = True):
    """Plot learning curves for different configurations."""
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different world sizes
    world_sizes = sorted(df['world_size'].unique())
    
    for i, ws in enumerate(world_sizes):
        ws_data = df[df['world_size'] == ws]
        
        plt.subplot(2, 2, i+1)
        
        # Plot curves for different learning rates
        for lr in sorted(ws_data['learning_rate'].unique()):
            lr_data = ws_data[ws_data['learning_rate'] == lr]
            
            # Average across seeds
            avg_data = lr_data.groupby('round')['test_accuracy'].mean()
            std_data = lr_data.groupby('round')['test_accuracy'].std()
            
            plt.plot(avg_data.index, avg_data.values, label=f'LR={lr}', marker='o')
            plt.fill_between(avg_data.index, 
                           avg_data.values - std_data.values, 
                           avg_data.values + std_data.values, 
                           alpha=0.2)
        
        plt.title(f'World Size = {ws}')
        plt.xlabel('Round')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_hyperparameter_comparison(df: pd.DataFrame, save_plots: bool = True):
    """Plot comparison of hyperparameters on final accuracy."""
    final_df = df.groupby(['world_size', 'learning_rate', 'enable_swapping', 'rounds', 'seed']).last()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Learning rate comparison
    sns.boxplot(data=final_df.reset_index(), x='learning_rate', y='test_accuracy', ax=axes[0,0])
    axes[0,0].set_title('Final Accuracy vs Learning Rate')
    axes[0,0].set_xlabel('Learning Rate')
    axes[0,0].set_ylabel('Test Accuracy')
    
    # World size comparison
    sns.boxplot(data=final_df.reset_index(), x='world_size', y='test_accuracy', ax=axes[0,1])
    axes[0,1].set_title('Final Accuracy vs World Size')
    axes[0,1].set_xlabel('World Size')
    axes[0,1].set_ylabel('Test Accuracy')
    
    # Swapping comparison
    sns.boxplot(data=final_df.reset_index(), x='enable_swapping', y='test_accuracy', ax=axes[1,0])
    axes[1,0].set_title('Final Accuracy vs Enable Swapping')
    axes[1,0].set_xlabel('Enable Swapping')
    axes[1,0].set_ylabel('Test Accuracy')
    
    # Rounds comparison
    sns.boxplot(data=final_df.reset_index(), x='rounds', y='test_accuracy', ax=axes[1,1])
    axes[1,1].set_title('Final Accuracy vs Number of Rounds')
    axes[1,1].set_xlabel('Rounds')
    axes[1,1].set_ylabel('Test Accuracy')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_best_configurations(df: pd.DataFrame, top_n: int = 5):
    """Print the best configurations based on final accuracy."""
    final_df = df.groupby(['world_size', 'learning_rate', 'enable_swapping', 'rounds', 'seed']).last()
    
    # Calculate mean accuracy for each configuration (averaging across seeds)
    config_means = final_df.reset_index().groupby(['world_size', 'learning_rate', 'enable_swapping', 'rounds'])['test_accuracy'].mean().sort_values(ascending=False)
    
    print(f"\nTop {top_n} Configurations by Final Accuracy:")
    print("=" * 60)
    
    for i, (config, accuracy) in enumerate(config_means.head(top_n).items()):
        world_size, learning_rate, enable_swapping, rounds = config
        print(f"{i+1}. World Size: {world_size}, LR: {learning_rate}, Swap: {enable_swapping}, "
              f"Rounds: {rounds}, Accuracy: {accuracy:.4f}")

def plot_specific_experiments(df: pd.DataFrame, 
                            world_sizes: List[int] = None,
                            learning_rates: List[float] = None,
                            enable_swapping: List[bool] = None,
                            rounds_list: List[int] = None,
                            seeds: List[int] = None,
                            save_plots: bool = True,
                            plot_title: str = "Learning Curves for Selected Experiments"):
    """
    Plot learning curves for experiments matching the specified criteria.
    
    Args:
        df: DataFrame containing experiment results
        world_sizes: List of world sizes to include (None means all)
        learning_rates: List of learning rates to include (None means all)
        enable_swapping: List of swapping configs to include (None means all)
        rounds_list: List of round counts to include (None means all)
        seeds: List of seeds to include (None means all)
        save_plots: Whether to save the plot to file
        plot_title: Title for the plot
    """
    # Filter the dataframe based on specified criteria
    filtered_df = df.copy()
    
    if world_sizes is not None:
        filtered_df = filtered_df[filtered_df['world_size'].isin(world_sizes)]
    if learning_rates is not None:
        filtered_df = filtered_df[filtered_df['learning_rate'].isin(learning_rates)]
    if enable_swapping is not None:
        filtered_df = filtered_df[filtered_df['enable_swapping'].isin(enable_swapping)]
    if rounds_list is not None:
        filtered_df = filtered_df[filtered_df['rounds'].isin(rounds_list)]
    if seeds is not None:
        filtered_df = filtered_df[filtered_df['seed'].isin(seeds)]
    
    if len(filtered_df) == 0:
        print("No experiments match the specified criteria.")
        return
    
    # Get unique configurations
    unique_configs = filtered_df[['world_size', 'learning_rate', 'enable_swapping', 'rounds']].drop_duplicates()
    
    print(f"Found {len(unique_configs)} unique configurations matching criteria:")
    for _, config in unique_configs.iterrows():
        matching_seeds = sorted(filtered_df[
            (filtered_df['world_size'] == config['world_size']) &
            (filtered_df['learning_rate'] == config['learning_rate']) &
            (filtered_df['enable_swapping'] == config['enable_swapping']) &
            (filtered_df['rounds'] == config['rounds'])
        ]['seed'].unique())
        print(f"  WS={config['world_size']}, LR={config['learning_rate']}, "
              f"Swap={config['enable_swapping']}, R={config['rounds']}, Seeds={matching_seeds}")
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Use different colors and markers for different configurations
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_configs)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, (_, config) in enumerate(unique_configs.iterrows()):
        config_data = filtered_df[
            (filtered_df['world_size'] == config['world_size']) &
            (filtered_df['learning_rate'] == config['learning_rate']) &
            (filtered_df['enable_swapping'] == config['enable_swapping']) &
            (filtered_df['rounds'] == config['rounds'])
        ]
        
        # Calculate mean and std across seeds for this configuration
        mean_data = config_data.groupby('round')['test_accuracy'].mean()
        std_data = config_data.groupby('round')['test_accuracy'].std()
        
        label = f"WS={config['world_size']}, LR={config['learning_rate']}, Swap={config['enable_swapping']}"
        
        # Plot mean with error bars
        plt.plot(mean_data.index, mean_data.values, 
                label=label, 
                color=colors[i], 
                marker=markers[i % len(markers)],
                linewidth=2, 
                markersize=6)
        
        # Add error bars (standard deviation)
        plt.fill_between(mean_data.index, 
                        mean_data.values - std_data.fillna(0).values, 
                        mean_data.values + std_data.fillna(0).values, 
                        color=colors[i], 
                        alpha=0.2)
    
    plt.title(plot_title, fontsize=16, fontweight='bold')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plots:
        # Create a filename based on the filters
        filename_parts = []
        if world_sizes is not None:
            filename_parts.append(f"ws{'-'.join(map(str, world_sizes))}")
        if learning_rates is not None:
            filename_parts.append(f"lr{'-'.join([str(lr).replace('.', '') for lr in learning_rates])}")
        if enable_swapping is not None:
            filename_parts.append(f"swap{'-'.join(map(str, enable_swapping))}")
        if rounds_list is not None:
            filename_parts.append(f"r{'-'.join(map(str, rounds_list))}")
        if seeds is not None:
            filename_parts.append(f"seed{'-'.join(map(str, seeds))}")
        
        filename = f"learning_curves_{'_'.join(filename_parts) if filename_parts else 'all'}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
    
    plt.show()

def generate_summary_report(results_file: str = "experiment_results.pkl"):
    """Generate a comprehensive summary report."""
    print("CIFAR100 AlexNet Experiment Results Analysis")
    print("=" * 50)
    
    # Load results
    results = load_experiment_results(results_file)
    if not results:
        print("No results to analyze.")
        return
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    print(f"\nTotal experiments: {len(results)}")
    print(f"Total data points: {len(df)}")
    
    # Basic statistics
    print(f"\nConfiguration ranges:")
    print(f"World sizes: {sorted(df['world_size'].unique())}")
    print(f"Learning rates: {sorted(df['learning_rate'].unique())}")
    print(f"Enable swapping: {sorted(df['enable_swapping'].unique())}")
    print(f"Rounds: {sorted(df['rounds'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    
    # Final results analysis
    print("\nFinal Results Statistics:")
    final_stats = analyze_final_results(df)
    
    # Set pandas display options to show full table
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
        print(final_stats)
    
    # Best configurations
    print_best_configurations(df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_learning_curves(df)
    plot_hyperparameter_comparison(df)
    
    print("\nAnalysis complete! Check the generated plots.")

def plot_experiments_interactive():
    """Interactive function to plot specific experiments."""
    results_file = "experiment_results.pkl"
    
    # Load results
    results = load_experiment_results(results_file)
    if not results:
        print("No results to analyze.")
        return
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    print("Available experiment parameters:")
    print(f"World sizes: {sorted(df['world_size'].unique())}")
    print(f"Learning rates: {sorted(df['learning_rate'].unique())}")
    print(f"Enable swapping: {sorted(df['enable_swapping'].unique())}")
    print(f"Rounds: {sorted(df['rounds'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    
    print("\nEnter comma-separated values for each parameter you want to filter by.")
    print("Press Enter to include all values for that parameter.")
    
    # Get user input
    world_sizes_input = input("World sizes (e.g., 2,4,8): ").strip()
    world_sizes = [int(x.strip()) for x in world_sizes_input.split(',')] if world_sizes_input else None
    
    learning_rates_input = input("Learning rates (e.g., 0.0001,0.001): ").strip()
    learning_rates = [float(x.strip()) for x in learning_rates_input.split(',')] if learning_rates_input else None
    
    swapping_input = input("Enable swapping (True,False): ").strip()
    enable_swapping = None
    if swapping_input:
        enable_swapping = [x.strip().lower() == 'true' for x in swapping_input.split(',')]
    
    rounds_input = input("Rounds (e.g., 10,20): ").strip()
    rounds_list = [int(x.strip()) for x in rounds_input.split(',')] if rounds_input else None
    
    seeds_input = input("Seeds (e.g., 42,123): ").strip()
    seeds = [int(x.strip()) for x in seeds_input.split(',')] if seeds_input else None
    
    plot_title = input("Plot title (or press Enter for default): ").strip()
    if not plot_title:
        plot_title = "Learning Curves for Selected Experiments"
    
    # Plot the selected experiments
    plot_specific_experiments(df, world_sizes, learning_rates, enable_swapping, 
                            rounds_list, seeds, save_plots=True, plot_title=plot_title)

def main():
    """Main function for the analysis script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze CIFAR100 AlexNet experiment results')
    parser.add_argument('--results-file', default='experiment_results.pkl', 
                       help='Path to the results pickle file')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip generating plots')
    parser.add_argument('--plot-specific', action='store_true',
                       help='Interactively select specific experiments to plot')
    
    args = parser.parse_args()
    
    # Change to the results directory
    base_dir = "/home/cc/flame/lib/python/examples/cifar100_alexnet"
    if os.path.exists(base_dir):
        os.chdir(base_dir)
    
    results_file = args.results_file
    
    if args.plot_specific:
        # Interactive plotting mode
        plot_experiments_interactive()
    elif args.no_plots:
        # Just load and print statistics
        results = load_experiment_results(results_file)
        if results:
            df = results_to_dataframe(results)
            print_best_configurations(df)
            final_stats = analyze_final_results(df)
            print("\nFinal Results Statistics:")
            # Set pandas display options to show full table
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
                print(final_stats)
    else:
        # Generate full report
        generate_summary_report(results_file)


    # PLOT SPECIFIC PLOTS
    df = results_to_dataframe(results)
    plot_specific_experiments(df, world_sizes=[2], learning_rates=[0.0001], enable_swapping=[False], seeds=[42])

# Example usage functions
def plot_by_world_size(world_sizes: List[int], results_file: str = "experiment_results.pkl"):
    """Plot learning curves comparing different world sizes."""
    results = load_experiment_results(results_file)
    if not results:
        return
    df = results_to_dataframe(results)
    plot_specific_experiments(df, world_sizes=world_sizes, 
                            plot_title=f"Learning Curves: World Sizes {world_sizes}")

def plot_by_learning_rate(learning_rates: List[float], results_file: str = "experiment_results.pkl"):
    """Plot learning curves comparing different learning rates."""
    results = load_experiment_results(results_file)
    if not results:
        return
    df = results_to_dataframe(results)
    plot_specific_experiments(df, learning_rates=learning_rates,
                            plot_title=f"Learning Curves: Learning Rates {learning_rates}")

def plot_swapping_comparison(world_sizes: List[int] = None, results_file: str = "experiment_results.pkl"):
    """Plot learning curves comparing with and without swapping."""
    results = load_experiment_results(results_file)
    if not results:
        return
    df = results_to_dataframe(results)
    plot_specific_experiments(df, world_sizes=world_sizes, enable_swapping=[True, False],
                            plot_title="Learning Curves: Swapping vs No Swapping")

if __name__ == "__main__":
    main()

