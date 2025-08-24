#!/usr/bin/env python3
"""
Analysis script for Llama 3.2 1B experiment results.

This script loads and analyzes the experiment results saved in pickle format.
"""
import os
import sys
import pickle
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Any


def load_experiment_results(results_file: str = "experiment_results_llama.pkl") -> Dict:
    """Load experiment results from pickle file."""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return {}

    try:
        with open(results_file, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)} experiment results")
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return {}


def results_to_dataframe(results_dict: Dict) -> pd.DataFrame:
    """Convert results dictionary to pandas DataFrame for analysis."""
    rows = []

    for key, experiment_results in results_dict.items():
        world_size, learning_rate, enable_swapping, rounds, epochs, dataset, seed = key

        for round_num, test_loss, perplexity in experiment_results:
            rows.append(
                {
                    "world_size": world_size,
                    "learning_rate": learning_rate,
                    "enable_swapping": enable_swapping,
                    "rounds": rounds,
                    "epochs": epochs,
                    "dataset": dataset,
                    "seed": seed,
                    "round": round_num,
                    "perplexity": perplexity,
                    "test_loss": test_loss,
                }
            )

    return pd.DataFrame(rows)


def analyze_final_results(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze final results for each experiment configuration."""
    # Get the final round for each experiment
    final_results = df.groupby(
        ["world_size", "learning_rate", "enable_swapping", "rounds", "epochs", "dataset", "seed"]
    ).last()

    # Calculate statistics
    stats = (
        final_results.groupby(
            ["world_size", "learning_rate", "enable_swapping", "rounds", "epochs", "dataset"]
        )
        .agg(
            {
                "test_loss": ["mean", "std", "min", "max"],
                "perplexity": ["mean", "std", "min", "max"],
            }
        )
        .round(4)
    )

    return stats


def plot_learning_curves(df: pd.DataFrame, save_plots: bool = True):
    """Plot learning curves for different configurations."""
    plt.figure(figsize=(20, 15))

    # Create subplots for different combinations
    datasets = sorted(df["dataset"].unique())
    world_sizes = sorted(df["world_size"].unique())
    
    subplot_idx = 1
    for dataset in datasets:
        for ws in world_sizes:
            plt.subplot(len(datasets), len(world_sizes), subplot_idx)
            subset_data = df[(df["dataset"] == dataset) & (df["world_size"] == ws)]
            
            if len(subset_data) == 0:
                subplot_idx += 1
                continue

            # Plot different learning rates and swapping configs
            for lr in sorted(subset_data['learning_rate'].unique()):
                for enable_swapping in sorted(subset_data['enable_swapping'].unique()):
                    # Don't plot swap=True for ws 1
                    if ws == 1 and enable_swapping: 
                        continue

                    subset = subset_data[(subset_data['learning_rate'] == lr) &
                                       (subset_data['enable_swapping'] == enable_swapping)]
                    if len(subset) > 0:
                        # Average across seeds (though we only have one seed)
                        avg_data = subset.groupby('round')['perplexity'].mean()
                        std_data = subset.groupby('round')['perplexity'].std()

                        label = f"lr={lr}, swap={enable_swapping}"
                        plt.plot(avg_data.index, avg_data.values, label=label, marker='o')
                        if not std_data.isna().all():
                            plt.fill_between(avg_data.index,
                                           avg_data.values - std_data.values,
                                           avg_data.values + std_data.values,
                                           alpha=0.2)

            plt.title(f"Dataset: {dataset}, World Size: {ws}")
            plt.xlabel("Round")
            plt.ylabel("Perplexity")
            
            # Limit y-axis for poems.txt to improve readability
            if "poems.txt" in dataset:
                plt.ylim(top=15, bottom=0)
                
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            subplot_idx += 1

    plt.tight_layout()
    if save_plots:
        plt.savefig("learning_curves_llama32_1b.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_hyperparameter_comparison(df: pd.DataFrame, save_plots: bool = True):
    """Plot comparison of hyperparameters on final perplexity."""
    final_df = df.groupby(
        ["world_size", "learning_rate", "enable_swapping", "rounds", "epochs", "dataset", "seed"]
    ).last()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Learning rate comparison
    sns.boxplot(
        data=final_df.reset_index(), x="learning_rate", y="perplexity", ax=axes[0], showfliers=False
    )
    axes[0].set_title("Perplexity vs Learning Rate")
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Perplexity")
    axes[0].tick_params(axis='x', rotation=45)

    # World size comparison
    sns.boxplot(
        data=final_df.reset_index(), x="world_size", y="perplexity", ax=axes[1], showfliers=False
    )
    axes[1].set_title("Perplexity vs World Size")
    axes[1].set_xlabel("World Size")
    axes[1].set_ylabel("Perplexity")

    # Enable swapping comparison
    sns.boxplot(
        data=final_df.reset_index(),
        x="enable_swapping",
        y="perplexity",
        ax=axes[2],
        showfliers=False
    )
    axes[2].set_title("Perplexity vs Enable Swapping")
    axes[2].set_xlabel("Enable Swapping")
    axes[2].set_ylabel("Perplexity")

    plt.tight_layout()
    if save_plots:
        plt.savefig(
            "hyperparameter_comparison_llama32_1b.png", dpi=300, bbox_inches="tight"
        )
    plt.show()


def print_best_configurations(df: pd.DataFrame, top_n: int = 5):
    """Print the best configurations based on final perplexity (lower is better)."""
    final_df = df.groupby(
        ["world_size", "learning_rate", "enable_swapping", "rounds", "epochs", "dataset", "seed"]
    ).last()

    # Calculate mean perplexity for each configuration (averaging across seeds)
    config_means = (
        final_df.reset_index()
        .groupby(
            ["world_size", "learning_rate", "enable_swapping", "rounds", "epochs", "dataset"]
        )["perplexity"]
        .mean()
        .sort_values(ascending=True)  # Lower perplexity is better
    )

    print(f"\nTop {top_n} Configurations by Final Perplexity (Lower is Better):")
    print("=" * 100)

    for i, (config, perplexity) in enumerate(config_means.head(top_n).items()):
        world_size, learning_rate, enable_swapping, rounds, epochs, dataset = config
        print(
            f"{i+1}. WS={world_size}, LR={learning_rate}, Swap={enable_swapping}, "
            f"Rounds={rounds}, Epochs={epochs}, Dataset={dataset}: {perplexity:.4f}"
        )


def plot_specific_experiments(
    df: pd.DataFrame,
    world_sizes: List[int] = None,
    learning_rates: List[float] = None,
    enable_swapping: List[bool] = None,
    rounds_list: List[int] = None,
    epochs_list: List[int] = None,
    datasets: List[str] = None,
    seeds: List[int] = None,
    save_plots: bool = True,
    plot_title: str = "Learning Curves for Selected Experiments",
):
    """
    Plot learning curves for experiments matching the specified criteria.
    """
    # Filter the dataframe based on specified criteria
    filtered_df = df.copy()

    if world_sizes is not None:
        filtered_df = filtered_df[filtered_df["world_size"].isin(world_sizes)]
    if learning_rates is not None:
        filtered_df = filtered_df[filtered_df["learning_rate"].isin(learning_rates)]
    if enable_swapping is not None:
        filtered_df = filtered_df[filtered_df["enable_swapping"].isin(enable_swapping)]
    if rounds_list is not None:
        filtered_df = filtered_df[filtered_df["rounds"].isin(rounds_list)]
    if epochs_list is not None:
        filtered_df = filtered_df[filtered_df["epochs"].isin(epochs_list)]
    if datasets is not None:
        filtered_df = filtered_df[filtered_df["dataset"].isin(datasets)]
    if seeds is not None:
        filtered_df = filtered_df[filtered_df["seed"].isin(seeds)]

    if len(filtered_df) == 0:
        print("No experiments found matching the specified criteria.")
        return

    # Get unique configurations
    unique_configs = filtered_df[
        ["world_size", "learning_rate", "enable_swapping", "rounds", "epochs", "dataset"]
    ].drop_duplicates()

    print(f"Found {len(unique_configs)} unique configurations matching criteria:")
    for _, config in unique_configs.iterrows():
        print(
            f"  WS={config['world_size']}, LR={config['learning_rate']}, Swap={config['enable_swapping']}, "
            f"Rounds={config['rounds']}, Epochs={config['epochs']}, Dataset={config['dataset']}"
        )

    # Create the plot
    plt.figure(figsize=(15, 10))

    # Use different colors and markers for different configurations
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_configs)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    for i, (_, config) in enumerate(unique_configs.iterrows()):
        config_data = filtered_df[
            (filtered_df["world_size"] == config["world_size"])
            & (filtered_df["learning_rate"] == config["learning_rate"])
            & (filtered_df["enable_swapping"] == config["enable_swapping"])
            & (filtered_df["rounds"] == config["rounds"])
            & (filtered_df["epochs"] == config["epochs"])
            & (filtered_df["dataset"] == config["dataset"])
        ]

        # Average across seeds (though we only have one seed)
        avg_data = config_data.groupby("round")["perplexity"].mean()
        std_data = config_data.groupby("round")["perplexity"].std()

        label = f"WS={config['world_size']}, LR={config['learning_rate']}, Swap={config['enable_swapping']}, R={config['rounds']}, E={config['epochs']}, D={config['dataset']}"
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(
            avg_data.index,
            avg_data.values,
            label=label,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=6,
        )
        if not std_data.isna().all():
            plt.fill_between(
                avg_data.index,
                avg_data.values - std_data.values,
                avg_data.values + std_data.values,
                alpha=0.2,
                color=color,
            )

    plt.title(plot_title, fontsize=16, fontweight="bold")
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_plots:
        # Create a safe filename from the plot title
        safe_filename = "".join(
            c for c in plot_title if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_filename = safe_filename.replace(" ", "_").lower()
        plt.savefig(f"{safe_filename}_llama32_1b.png", dpi=300, bbox_inches="tight")

    plt.show()


def generate_summary_report(results_file: str = "experiment_results_llama.pkl"):
    """Generate a comprehensive summary report."""
    print("Llama 3.2 1B Experiment Results Analysis")
    print("=" * 50)

    # Load results
    results = load_experiment_results(results_file)
    if not results:
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
    print(f"Epochs: {sorted(df['epochs'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    # Final results analysis
    print("\nFinal Results Statistics:")
    final_stats = analyze_final_results(df)

    # Set pandas display options to show full table
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        None,
    ):
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
    results_file = "experiment_results_llama32.pkl"

    # Load results
    results = load_experiment_results(results_file)
    if not results:
        return

    # Convert to DataFrame
    df = results_to_dataframe(results)

    print("Available experiment parameters:")
    print(f"World sizes: {sorted(df['world_size'].unique())}")
    print(f"Learning rates: {sorted(df['learning_rate'].unique())}")
    print(f"Enable swapping: {sorted(df['enable_swapping'].unique())}")
    print(f"Rounds: {sorted(df['rounds'].unique())}")
    print(f"Epochs: {sorted(df['epochs'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    print("\nEnter comma-separated values for each parameter you want to filter by.")
    print("Press Enter to include all values for that parameter.")

    # Get user input
    world_sizes_input = input("World sizes (e.g., 1,2): ").strip()
    world_sizes = (
        [int(x.strip()) for x in world_sizes_input.split(",")]
        if world_sizes_input
        else None
    )

    learning_rates_input = input("Learning rates (e.g., 1e-6,1e-5): ").strip()
    learning_rates = (
        [float(x.strip()) for x in learning_rates_input.split(",")]
        if learning_rates_input
        else None
    )

    swapping_input = input("Enable swapping (True,False): ").strip()
    enable_swapping = None
    if swapping_input:
        enable_swapping = [
            x.strip().lower() == "true" for x in swapping_input.split(",")
        ]

    rounds_input = input("Rounds (e.g., 3,5,8): ").strip()
    rounds_list = (
        [int(x.strip()) for x in rounds_input.split(",")] if rounds_input else None
    )

    epochs_input = input("Epochs (e.g., 1,2,3): ").strip()
    epochs_list = (
        [int(x.strip()) for x in epochs_input.split(",")] if epochs_input else None
    )

    datasets_input = input("Datasets (e.g., simple_math.txt,random.txt): ").strip()
    datasets = (
        [x.strip() for x in datasets_input.split(",")] if datasets_input else None
    )

    seeds_input = input("Seeds (e.g., 42): ").strip()
    seeds = [int(x.strip()) for x in seeds_input.split(",")] if seeds_input else None

    plot_title = input("Plot title (or press Enter for default): ").strip()
    if not plot_title:
        plot_title = "Learning Curves for Selected Experiments"

    # Plot the selected experiments
    plot_specific_experiments(
        df,
        world_sizes,
        learning_rates,
        enable_swapping,
        rounds_list,
        epochs_list,
        datasets,
        seeds,
        save_plots=True,
        plot_title=plot_title,
    )


def main():
    """Main function for the analysis script."""

    parser = argparse.ArgumentParser(
        description="Analyze Llama 3.2 1B experiment results"
    )
    parser.add_argument(
        "--results-file",
        default="aggregator/experiment_results_llama32.pkl",
        help="Path to the results pickle file",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--plot-specific",
        action="store_true",
        help="Interactively select specific experiments to plot",
    )

    args = parser.parse_args()

    # Change to the results directory
    base_dir = "/home/cc/flame/lib/python/examples/llama_3.2_1B"
    if os.path.exists(base_dir):
        os.chdir(base_dir)

    results_file = args.results_file

    if args.plot_specific:
        plot_experiments_interactive()
    elif args.no_plots:
        results = load_experiment_results(results_file)

        if results:
            df = results_to_dataframe(results)
            print_best_configurations(df)
    else:
        generate_summary_report(results_file)


# Example usage functions
def plot_by_world_size(
    world_sizes: List[int], results_file: str = "experiment_results_llama.pkl"
):
    """Plot learning curves comparing different world sizes."""
    results = load_experiment_results(results_file)
    if not results:
        return
    df = results_to_dataframe(results)
    plot_specific_experiments(
        df,
        world_sizes=world_sizes,
        plot_title=f"Learning Curves: World Sizes {world_sizes}",
    )


def plot_by_learning_rate(
    learning_rates: List[float], results_file: str = "experiment_results_llama.pkl"
):
    """Plot learning curves comparing different learning rates."""
    results = load_experiment_results(results_file)
    if not results:
        return
    df = results_to_dataframe(results)
    plot_specific_experiments(
        df,
        learning_rates=learning_rates,
        plot_title=f"Learning Curves: Learning Rates {learning_rates}",
    )


def plot_by_dataset(
    datasets: List[str], results_file: str = "experiment_results_llama.pkl"
):
    """Plot learning curves comparing different datasets."""
    results = load_experiment_results(results_file)
    if not results:
        return
    df = results_to_dataframe(results)
    plot_specific_experiments(
        df,
        datasets=datasets,
        plot_title=f"Learning Curves: Datasets {datasets}",
    )


def plot_swapping_comparison(
    world_sizes: List[int] = None, results_file: str = "experiment_results_llama.pkl"
):
    """Plot learning curves comparing with and without swapping."""
    results = load_experiment_results(results_file)
    if not results:
        return
    df = results_to_dataframe(results)
    plot_specific_experiments(
        df,
        world_sizes=world_sizes,
        enable_swapping=[True, False],
        plot_title=f"Learning Curves: Swapping Comparison",
    )


if __name__ == "__main__":
    main()
