#!/usr/bin/env python3
"""
Analysis script for CIFAR100 AlexNet experiment results.

This script loads and analyzes the experiment results saved in pickle format.
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
    print(final_stats)
    
    # Best configurations
    print_best_configurations(df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_learning_curves(df)
    plot_hyperparameter_comparison(df)
    
    print("\nAnalysis complete! Check the generated plots.")

def main():
    """Main function for the analysis script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze CIFAR100 AlexNet experiment results')
    parser.add_argument('--results-file', default='experiment_results.pkl', 
                       help='Path to the results pickle file')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Change to the results directory
    base_dir = "/home/cc/flame/lib/python/examples/cifar100_alexnet"
    if os.path.exists(base_dir):
        os.chdir(base_dir)
    
    results_file = args.results_file
    
    if args.no_plots:
        # Just load and print statistics
        results = load_experiment_results(results_file)
        if results:
            df = results_to_dataframe(results)
            print_best_configurations(df)
            final_stats = analyze_final_results(df)
            print("\nFinal Results Statistics:")
            print(final_stats)
    else:
        # Generate full report
        generate_summary_report(results_file)

if __name__ == "__main__":
    main()
