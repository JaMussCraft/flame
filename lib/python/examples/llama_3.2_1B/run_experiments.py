#!/usr/bin/env python3
"""
Automation script for running Llama 3.2 1B experiments with different configurations.

This script runs experiments with different combinations of hyperparameters and saves
results to a pickle file for later analysis.
"""
import os
import sys
import json
import time
import pickle
import signal
import logging
import subprocess
from datetime import datetime
from itertools import product
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaExperimentRunner:
    """Class to manage and run Llama 3.2 1B experiments."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.aggregator_dir = os.path.join(base_dir, "aggregator")
        self.trainer_dir = os.path.join(base_dir, "trainer")
        self.results_file = os.path.join(base_dir, "experiment_results_llama.pkl")
        self.results_dict = {}
        
        # Experiment configuration lists - adapted for Llama
        self.world_sizes = [1, 2]
        self.learning_rates = [1e-6, 1e-5, 1e-4]
        self.datasets = ["simple_math.txt", "poems.txt", "qa_pairs.txt"]  # Different datasets to finetune on
        self.trainer_epochs = [2]  # Different epochs for trainer
        # self.trainer_epochs = [1, 2, 3]  # Different epochs for trainer
        self.aggregator_rounds = [5]  # Different rounds for aggregator
        # self.aggregator_rounds = [3, 5, 8]  # Different rounds for aggregator
        self.enable_swapping_configs = [False, True]
        self.seeds = [42]  # Single seed since no random initialization
        
        # Load existing results if available
        self._load_existing_results()
    
    def _load_existing_results(self):
        """Load existing experiment results if available."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'rb') as f:
                    self.results_dict = pickle.load(f)
                logger.info(f"Loaded {len(self.results_dict)} existing experiment results")
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
                self.results_dict = {}
    
    def _save_results(self):
        """Save current results to pickle file."""
        try:
            with open(self.results_file, 'wb') as f:
                pickle.dump(self.results_dict, f)
            logger.info(f"Saved results to {self.results_file}")
        except Exception as e:
            logger.error(f"Could not save results: {e}")
    
    def _create_aggregator_config(self, world_size: int, seed: int, enable_swapping: bool, 
                                 rounds: int, learning_rate: float, dataset: str) -> str:
        """Create aggregator config file with specified parameters."""
        config = {
            "taskid": "llama32_1b_aggregator_001",
            "backend": "p2p",
            "brokers": [
                {
                    "host": "localhost",
                    "sort": "mqtt"
                },
                {
                    "host": "localhost:10104",
                    "sort": "p2p"
                }
            ],
            "groupAssociation": {
                "param-channel": "default"
            },
            "channels": [
                {
                    "description": "Model update is sent from trainer to aggregator and vice-versa",
                    "groupBy": {
                        "type": "tag",
                        "value": [
                            "default"
                        ]
                    },
                    "name": "param-channel",
                    "pair": [
                        "trainer",
                        "aggregator"
                    ],
                    "funcTags": {
                        "aggregator": [
                            "distribute",
                            "aggregate"
                        ],
                        "trainer": [
                            "fetch",
                            "upload"
                        ]
                    }
                }
            ],
            "dataset": "custom_text",
            "dependencies": [
                "numpy >= 1.2.0",
                "torch >= 1.12.0",
                "tiktoken"
            ],
            "hyperparameters": {
                "rounds": rounds,
                "world_size": world_size,
                "seed": seed,
                "learning_rate": learning_rate,
                "batch_size": 4,
                "max_seq_len": 512,
                "ckpt_dir": "../checkpoints/pretrained",
                "enable_swapping": enable_swapping,
                "data_path": f"test_dataset/{dataset}"
            },
            "baseModel": {
                "name": "llama-3.2-1b",
                "version": 1
            },
            "job": {
                "id": "622a358619ab59012eabeefb",
                "name": "llama32_1b_federated"
            },
            "registry": {
                "sort": "dummy",
                "uri": ""
            },
            "selector": {
                "sort": "default",
                "k-args": {}
            },
            "optimizer": {
                "sort": "fedavg",
                "kwargs": {}
            },
            "maxRunTime": 3600,
            "realm": "default",
            "role": "aggregator"
        }
        
        config_path = os.path.join(self.aggregator_dir, "config_temp.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    def _create_trainer_config(self, trainer_id: int, world_size: int, learning_rate: float, 
                             rank: int, seed: int, epochs: int, dataset: str) -> str:
        """Create trainer config file with specified parameters."""
        config = {
            "taskid": f"llama32_1b_trainer_{trainer_id:03d}",
            "backend": "p2p",
            "brokers": [
                {
                    "host": "localhost",
                    "sort": "mqtt"
                },
                {
                    "host": "localhost:10104",
                    "sort": "p2p"
                }
            ],
            "groupAssociation": {
                "param-channel": "default"
            },
            "channels": [
                {
                    "description": "Model update is sent from trainer to aggregator and vice-versa",
                    "groupBy": {
                        "type": "tag",
                        "value": ["default"]
                    },
                    "name": "param-channel",
                    "pair": ["trainer", "aggregator"],
                    "funcTags": {
                        "aggregator": ["distribute", "aggregate"],
                        "trainer": ["fetch", "upload"]
                    }
                }
            ],
            "dataset": "custom_text",
            "dependencies": ["numpy >= 1.2.0", "torch >= 1.12.0", "tiktoken"],
            "hyperparameters": {
                "batch_size": 4,
                "learning_rate": learning_rate,
                "rank": rank,
                "world_size": world_size,
                "epochs": epochs,
                "seed": seed,
                "max_seq_len": 512,
                "ckpt_dir": "../checkpoints/pretrained",
                "weight_decay": 0.01,
                "data_path": f"train_dataset/{dataset}"
            },
            "baseModel": {
                "name": "llama-3.2-1b",
                "version": 1
            },
            "job": {
                "id": "622a358619ab59012eabeefb",
                "name": "llama32_1b_federated"
            },
            "registry": {
                "sort": "dummy",
                "uri": ""
            },
            "selector": {
                "sort": "default",
                "kwargs": {}
            },
            "optimizer": {
                "sort": "fedavg",
                "kwargs": {}
            },
            "maxRunTime": 3600,
            "realm": "default/us/west",
            "role": "trainer"
        }
        
        config_path = os.path.join(self.trainer_dir, f"config_temp_{trainer_id}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    def _cleanup_temp_files(self):
        """Clean up temporary config files."""
        temp_files = [
            os.path.join(self.aggregator_dir, "config_temp.json"),
        ]
        
        # Add trainer temp files
        for i in range(1, 10):  # Assuming max 10 trainers
            temp_file = os.path.join(self.trainer_dir, f"config_temp_{i}.json")
            if os.path.exists(temp_file):
                temp_files.append(temp_file)
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Removed temp file: {temp_file}")
    
    def _run_single_experiment(self, world_size: int, learning_rate: float, 
                              enable_swapping: bool, rounds: int, epochs: int, 
                              dataset: str, seed: int) -> str:
        """Run a single experiment with specified parameters."""
        
        experiment_key = (world_size, learning_rate, enable_swapping, rounds, epochs, dataset, seed)
        
        # Check if experiment already exists
        if experiment_key in self.results_dict:
            logger.info(f"Experiment {experiment_key} already exists, skipping...")
            return "completed"
        
        logger.info(f"Starting experiment: world_size={world_size}, lr={learning_rate}, "
                   f"enable_swapping={enable_swapping}, rounds={rounds}, epochs={epochs}, "
                   f"dataset={dataset}, seed={seed}")
        
        processes = []
        log_files = []
        
        # Create logs directory for this experiment
        experiment_name = f"ws{world_size}_lr{learning_rate}_swap{enable_swapping}_r{rounds}_e{epochs}_{dataset.replace('.txt', '')}_seed{seed}"
        logs_dir = os.path.join(self.base_dir, "logs", experiment_name)
        os.makedirs(logs_dir, exist_ok=True)
        
        try:
            # Create aggregator config
            agg_config_path = self._create_aggregator_config(world_size, seed, enable_swapping, rounds, learning_rate, dataset)
            
            # Start metaserver
            logger.info("Starting metaserver...")
            metaserver_log = open(os.path.join(logs_dir, "metaserver.log"), 'w')
            log_files.append(metaserver_log)
            metaserver_proc = subprocess.Popen(
                ["sudo", "/home/cc/.flame/bin/metaserver"],
                stdout=metaserver_log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
            processes.append(metaserver_proc)
            time.sleep(3)  # Give metaserver time to start
            
            # Start aggregator
            logger.info("Starting aggregator...")
            aggregator_log = open(os.path.join(logs_dir, "aggregator.log"), 'w')
            log_files.append(aggregator_log)
            aggregator_proc = subprocess.Popen(
                ["python", os.path.join(self.aggregator_dir, "main.py"), agg_config_path],
                stdout=aggregator_log,
                stderr=subprocess.STDOUT,
                cwd=self.aggregator_dir,
                preexec_fn=os.setsid
            )
            processes.append(aggregator_proc)
            time.sleep(10)  # Give aggregator time to start
            
            # Start trainers
            trainer_procs = []
            for trainer_id in range(1, world_size + 1):
                logger.info(f"Starting trainer {trainer_id}...")
                trainer_config_path = self._create_trainer_config(trainer_id, world_size, learning_rate, trainer_id - 1, seed, epochs, dataset)
                trainer_log = open(os.path.join(logs_dir, f"trainer_{trainer_id}.log"), 'w')
                log_files.append(trainer_log)
                trainer_proc = subprocess.Popen(
                    ["python", os.path.join(self.trainer_dir, "main.py"), trainer_config_path],
                    stdout=trainer_log,
                    stderr=subprocess.STDOUT,
                    cwd=self.trainer_dir,
                    preexec_fn=os.setsid
                )
                trainer_procs.append(trainer_proc)
                time.sleep(2)  # Stagger trainer starts
            
            # Wait for all processes to complete (with timeout)
            logger.info("Waiting for experiment to complete...")
            logger.info(f"Process logs are being saved to: {logs_dir}")
            aggregator_proc.wait()

            # Wait for trainers to finish
            for i, trainer_proc in enumerate(trainer_procs):
                try:
                    trainer_proc.wait(timeout=10)
                    logger.info(f"Trainer {i+1} finished")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Trainer {i+1} did not finish in time, terminating...")
                    trainer_proc.terminate()
            
            # Record experiment completion
            self.results_dict[experiment_key] = "completed"
            self._save_results()
            
            logger.info(f"Experiment completed successfully: {experiment_key}")
            return "completed"
            
        except Exception as e:
            logger.error(f"Error running experiment {experiment_key}: {e}")
            self.results_dict[experiment_key] = f"error: {e}"
            self._save_results()
            return f"error: {e}"
        
        finally:
            # Close log files
            for log_file in log_files:
                log_file.close()

            # Clean up processes
            for proc in processes + trainer_procs:
                try:
                    if proc.poll() is None:  # Process is still running
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        proc.wait(timeout=5)
                except Exception as e:
                    logger.warning(f"Error terminating process: {e}")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except:
                        pass
            
            # Cleanup temp files
            self._cleanup_temp_files()
            
            # Give some time between experiments
            time.sleep(2)
    
    def run_all_experiments(self):
        """Run all experiment combinations."""
        # Generate all combinations
        all_combinations = list(product(
            self.world_sizes,
            self.learning_rates,
            self.enable_swapping_configs,
            self.aggregator_rounds,
            self.trainer_epochs,
            self.datasets,
            self.seeds
        ))
        
        logger.info(f"Total experiments to run: {len(all_combinations)}")
        
        for i, (world_size, learning_rate, enable_swapping, rounds, epochs, dataset, seed) in enumerate(all_combinations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {i+1}/{len(all_combinations)}")
            logger.info(f"world_size={world_size}, lr={learning_rate}, enable_swapping={enable_swapping}, "
                       f"rounds={rounds}, epochs={epochs}, dataset={dataset}, seed={seed}")
            logger.info(f"{'='*60}")

            if world_size == 1 and enable_swapping:
                logger.info("Skipping swapping experiment for world_size=1")
                continue
            
            self._run_single_experiment(world_size, learning_rate, enable_swapping, rounds, epochs, dataset, seed)
            
            # Small delay between experiments
            time.sleep(5)
        
        logger.info("\n=== All experiments completed ===")
        logger.info(f"Total experiments in results: {len(self.results_dict)}")
        
        # Print summary
        self._print_results_summary()
    
    def _print_results_summary(self):
        """Print a summary of all results."""
        logger.info("\n=== Results Summary ===")
        completed_experiments = 0
        for key, status in self.results_dict.items():
            if status == "completed":
                completed_experiments += 1
            else:
                logger.info(f"Experiment {key}: {status}")
        
        logger.info(f"\nTotal completed experiments: {completed_experiments}/{len(self.results_dict)}")
        logger.info("Detailed results are saved in experiment_results_llama.pkl by the aggregator.")
        logger.info("Process logs are saved in the 'logs' directory for each experiment.")
    
    def check_experiment_logs(self, experiment_key):
        """Check logs for a specific experiment to help debug issues."""
        world_size, learning_rate, enable_swapping, rounds, epochs, dataset, seed = experiment_key
        experiment_name = f"ws{world_size}_lr{learning_rate}_swap{enable_swapping}_r{rounds}_e{epochs}_{dataset.replace('.txt', '')}_seed{seed}"
        logs_dir = os.path.join(self.base_dir, "logs", experiment_name)
        
        if not os.path.exists(logs_dir):
            logger.info(f"No logs found for experiment {experiment_name}")
            return
        
        logger.info(f"\n=== Log Summary for {experiment_name} ===")
        log_files = ["metaserver.log", "aggregator.log"] + [f"trainer_{i}.log" for i in range(1, world_size + 1)]
        
        for log_file in log_files:
            log_path = os.path.join(logs_dir, log_file)
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    logger.info(f"{log_file}: {len(lines)} lines")
                    if lines:
                        logger.info(f"  Last line: {lines[-1].strip()}")
            else:
                logger.info(f"{log_file}: not found")


def main():
    """Main function to run the experiment automation."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python run_experiments.py [--help] [--check-logs]")
            print("  --help: Show this help message")
            print("  --check-logs: Check logs for failed experiments")
            return
        elif sys.argv[1] == "--check-logs":
            # TODO: Implement log checking functionality
            print("Log checking not yet implemented")
            return
    
    # Set up the experiment runner
    base_dir = "/home/cc/flame/lib/python/examples/llama_3.2_1B"
    
    if not os.path.exists(base_dir):
        logger.error(f"Base directory not found: {base_dir}")
        return
    
    runner = LlamaExperimentRunner(base_dir)
    
    try:
        runner.run_all_experiments()
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Experiment automation finished")


if __name__ == "__main__":
    main()
