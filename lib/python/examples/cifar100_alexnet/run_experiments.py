#!/usr/bin/env python3
"""
Automation script for running CIFAR100 AlexNet experiments with different configurations.

This script runs experiments with different combinations of hyperparameters and saves
results to a pickle file for later analysis.
"""

import os
import sys
import json
import pickle
import time
import subprocess
import signal
import logging
from datetime import datetime
from itertools import product
from typing import Dict, Any
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Class to manage and run CIFAR100 AlexNet experiments."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.aggregator_dir = os.path.join(base_dir, "aggregator")
        self.trainer_dir = os.path.join(base_dir, "trainer")
        self.results_file = os.path.join(base_dir, "experiment_results.pkl")
        self.results_dict = {}
        
        # Experiment configuration lists
        self.world_sizes = [2]
        self.learning_rates = [0.0001]
        self.swap_configs = [False]
        self.rounds = [10]
        self.seeds = [42]
        # self.world_sizes = [1,2,4,8]
        # self.learning_rates = [0.00001, 0.0001, 0.001]
        # self.swap_configs = [False, True]
        # self.rounds = [10]
        # self.seeds = [123,42,77]
        
        # Load existing results if available
        self._load_existing_results()
    
    def _load_existing_results(self):
        """Load existing experiment results if available."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'rb') as f:
                    self.results_dict = pickle.load(f)
                logger.info(f"Loaded existing results with {len(self.results_dict)} experiments")
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
    
    def _create_aggregator_config(self, world_size: int, seed: int, enable_swapping: bool, rounds: int, learning_rate: float) -> str:
        """Create aggregator config file with specified parameters."""
        config = {
            "taskid": "cifar100alexnet_aggregator_001",
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
            "dataset": "cifar100",
            "dependencies": [
                "numpy >= 1.2.0",
                "torchvision >= 0.8.0"
            ],
            "hyperparameters": {
                "rounds": rounds,
                "world_size": world_size,
                "seed": seed,
                "enable_swapping": enable_swapping,
                "enable_layerwise_swapping": True,
                "learningRate": learning_rate
            },
            "baseModel": {
                "name": "",
                "version": 1
            },
            "job": {
                "id": "622a358619ab59012eabeefb",
                "name": "cifar100_alexnet"
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
            "maxRunTime": 600,
            "realm": "default",
            "role": "aggregator"
        }
        
        config_path = os.path.join(self.aggregator_dir, "config_temp.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    def _create_trainer_config(self, trainer_id: int, world_size: int, learning_rate: float, 
                             rank: int, seed: int) -> str:
        """Create trainer config file with specified parameters."""
        config = {
            "taskid": f"cifar100alexnet_trainer_{trainer_id:03d}",
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
            "dataset": "cifar100",
            "dependencies": ["numpy >= 1.2.0", "torchvision >= 0.8.0"],
            "hyperparameters": {
                "batchSize": 32,
                "learningRate": learning_rate,
                "rank": rank,
                "world_size": world_size,
                "epochs": 3,
                "seed": seed
            },
            "baseModel": {
                "name": "",
                "version": 1
            },
            "job": {
                "id": "622a358619ab59012eabeefb",
                "name": "cifar100_alexnet"
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
            "maxRunTime": 600,
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
                try:
                    os.remove(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove {temp_file}: {e}")
    
    def _run_single_experiment(self, world_size: int, learning_rate: float, 
                              enable_swapping: bool, rounds: int, seed: int) -> str:
        """Run a single experiment with specified parameters."""
        
        experiment_key = (world_size, learning_rate, enable_swapping, rounds, seed)
        
        # Check if experiment already exists
        if experiment_key in self.results_dict:
            logger.info(f"Experiment {experiment_key} already exists, skipping...")
            return "completed"
        
        logger.info(f"Starting experiment: world_size={world_size}, lr={learning_rate}, "
                   f"swap={enable_swapping}, rounds={rounds}, seed={seed}")
        
        processes = []
        log_files = []
        
        # Create logs directory for this experiment
        experiment_name = f"ws{world_size}_lr{learning_rate}_swap{enable_swapping}_r{rounds[0] if isinstance(rounds, list) else rounds}_seed{seed}"
        logs_dir = os.path.join(self.base_dir, "logs", experiment_name)
        os.makedirs(logs_dir, exist_ok=True)
        
        try:
            # Create aggregator config
            agg_config_path = self._create_aggregator_config(world_size, seed, enable_swapping, rounds, learning_rate)
            
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
            time.sleep(5)  # Give aggregator time to start
            
            # Start trainers
            trainer_procs = []
            for trainer_id in range(1, world_size + 1):
                rank = trainer_id - 1  # ranks start from 0
                trainer_config_path = self._create_trainer_config(
                    trainer_id, world_size, learning_rate, rank, seed
                )
                
                logger.info(f"Starting trainer {trainer_id} (rank {rank})...")
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
                processes.append(trainer_proc)
                time.sleep(2)  # Stagger trainer starts
            
            # Wait for training to complete
            # Monitor the aggregator process
            logger.info("Waiting for training to complete...")
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
            
            # Results are already saved by the modified aggregator main.py
            # We just need to record that this experiment was completed
            self.results_dict[experiment_key] = "completed"
            self._save_results()
            
            logger.info(f"Experiment completed successfully: {experiment_key}")
            return "completed"
            
        except Exception as e:
            logger.error(f"Error running experiment {experiment_key}: {e}")
            return "failed"
        
        finally:
            # Close log files first
            for log_file in log_files:
                try:
                    log_file.close()
                except Exception as e:
                    logger.warning(f"Error closing log file: {e}")
            
            # Clean up processes
            for proc in processes:
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
            
            # Clean up temp files
            self._cleanup_temp_files()
            
            # Give some time between experiments
            time.sleep(5)
    
    def run_all_experiments(self):
        """Run all experiment combinations."""
        # Generate all combinations
        all_combinations = list(product(
            self.world_sizes,
            self.learning_rates,
            self.swap_configs,
            self.rounds,
            self.seeds
        ))
        
        logger.info(f"Total experiments to run: {len(all_combinations)}")
        
        for i, (world_size, learning_rate, enable_swapping, rounds, seed) in enumerate(all_combinations):
            logger.info(f"\n=== Running experiment {i+1}/{len(all_combinations)} ===")
            
            # Skip swapping experiments when world_size is 1
            if world_size == 1 and enable_swapping:
                logger.info("Skipping swapping experiment for world_size=1")
                continue
            
            try:
                self._run_single_experiment(world_size, learning_rate, enable_swapping, rounds, seed)
            except Exception as e:
                logger.error(f"Failed to run experiment {i+1}: {e}")
                continue
        
        logger.info("\n=== All experiments completed ===")
        logger.info(f"Total experiments in results: {len(self.results_dict)}")
        
        # Print summary
        self._print_results_summary()
    
    def _print_results_summary(self):
        """Print a summary of all results."""
        logger.info("\n=== Results Summary ===")
        completed_experiments = 0
        for key, status in self.results_dict.items():
            world_size, learning_rate, enable_swapping, rounds, seed = key
            if status == "completed":
                completed_experiments += 1
                logger.info(f"✓ WS={world_size}, LR={learning_rate}, Swap={enable_swapping}, "
                           f"R={rounds}, Seed={seed}: Completed")
            else:
                logger.info(f"✗ WS={world_size}, LR={learning_rate}, Swap={enable_swapping}, "
                           f"R={rounds}, Seed={seed}: Failed")
        
        logger.info(f"\nTotal completed experiments: {completed_experiments}/{len(self.results_dict)}")
        logger.info("Detailed results are saved in experiment_results.pkl by the aggregator.")
        logger.info("Process logs are saved in the 'logs' directory for each experiment.")
    
    def check_experiment_logs(self, experiment_key):
        """Check logs for a specific experiment to help debug issues."""
        world_size, learning_rate, enable_swapping, rounds, seed = experiment_key
        experiment_name = f"ws{world_size}_lr{learning_rate}_swap{enable_swapping}_r{rounds}_seed{seed}"
        logs_dir = os.path.join(self.base_dir, "logs", experiment_name)
        
        if not os.path.exists(logs_dir):
            logger.warning(f"No logs found for experiment: {experiment_name}")
            return
        
        logger.info(f"\n=== Log Summary for {experiment_name} ===")
        log_files = ["metaserver.log", "aggregator.log"] + [f"trainer_{i}.log" for i in range(1, world_size + 1)]
        
        for log_file in log_files:
            log_path = os.path.join(logs_dir, log_file)
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            logger.info(f"\n{log_file} (last 5 lines):")
                            for line in lines[-5:]:
                                logger.info(f"  {line.strip()}")
                        else:
                            logger.info(f"{log_file}: Empty")
                except Exception as e:
                    logger.warning(f"Could not read {log_file}: {e}")
            else:
                logger.info(f"{log_file}: Not found")


def main():
    """Main function to run the experiment automation."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python run_experiments.py [--check-logs]")
            print("This script runs CIFAR100 AlexNet experiments with different configurations.")
            print("Results are saved to experiment_results.pkl")
            print("Process logs are saved to logs/ directory for each experiment.")
            print("\nOptions:")
            print("  --check-logs    Check and display the last few lines of recent experiment logs")
            return
        elif sys.argv[1] == "--check-logs":
            base_dir = "/home/cc/flame/lib/python/examples/cifar100_alexnet"
            runner = ExperimentRunner(base_dir)
            
            # Check logs for the most recent experiments
            logs_base_dir = os.path.join(base_dir, "logs")
            if os.path.exists(logs_base_dir):
                recent_experiments = sorted(os.listdir(logs_base_dir))[-3:]  # Last 3 experiments
                for exp_dir in recent_experiments:
                    logger.info(f"\n=== Checking logs for {exp_dir} ===")
                    exp_path = os.path.join(logs_base_dir, exp_dir)
                    if os.path.isdir(exp_path):
                        log_files = os.listdir(exp_path)
                        for log_file in sorted(log_files):
                            log_path = os.path.join(exp_path, log_file)
                            try:
                                with open(log_path, 'r') as f:
                                    lines = f.readlines()
                                    if lines:
                                        print(f"\n{log_file} (last 3 lines):")
                                        for line in lines[-3:]:
                                            print(f"  {line.strip()}")
                                    else:
                                        print(f"{log_file}: Empty")
                            except Exception as e:
                                print(f"Could not read {log_file}: {e}")
            else:
                print("No logs directory found")
            return
    
    # Set up the experiment runner
    base_dir = "/home/cc/flame/lib/python/examples/cifar100_alexnet"
    
    if not os.path.exists(base_dir):
        logger.error(f"Base directory not found: {base_dir}")
        return
    
    runner = ExperimentRunner(base_dir)
    
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
