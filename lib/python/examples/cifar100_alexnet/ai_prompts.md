Good settings combinations:

- 1 trainer, seed = 123, lr = 0.001
- 2 trainers, seed = 123, lr = 0.0001



Write an automation script (possible to be in python?) to run cifar100_alexnet experiments with different configurations as background processes. The script and result file should be stored in /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet.


To run an experiment with one trainer, you will need to run the following as separate processes:
- Run meta server: sudo /home/cc/.flame/bin/metaserver
- Run aggregator: python /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/aggregator/main.py /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/aggregator/config.json
- Run trainer 1: python /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/trainer/main.py /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/trainer/config1.json

To run an experiment with one trainer, you will need to run the following as separate processes:
- Run meta server: sudo /home/cc/.flame/bin/metaserver
- Run aggregator: python /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/aggregator/main.py /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/aggregator/config.json
- Run trainer 1: python /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/trainer/main.py /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/trainer/config1.json
- Run trainer 2: python /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/trainer/main.py /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/trainer/config2.json

The only values you should modify in an aggregator config file are:
- hyperparameters.world_size (same as the number of trainers)
- hyperparameters.seed (note: the best seed for 1 trainer and 2 trainers is "123")
- hyperparameters.enable_swapping (true or false)

The only values you should modify in a trainer config file are:
- taskid (just increment 001 to 002 and so on for different trainers)
- hyperparameters.world_size (same as the number of trainers)
- hyperparameters.learningRate
- hyperparameters.rank (starts from 0 for trainer 1, 1 for trainer 2, and so on...)
- hyperparameters.seed (note: the best seed for 1 trainer and 2 trainers is "123")

During evaluation inside /home/cc/FedLora/Flame-Experiments/flame/lib/python/examples/cifar100_alexnet/aggregator/main.py, instead of writing the results to a file with its current way, I want to save the results dictionary using pickle such that it'll be easy to create any kind of graphs from it. The dictionary should map different tuples of experiment configurations to list of tuples of (round, test_loss, test_accuracy). I want all the results from different experiments to be saved into one single file. The dictionary key should be (world_size, learning_rate, swap, rounds, seed)


Do not worry about generating graphs from the saved result file for now. I'll work on it later.


I don't want to hard code different experiment configurations. I want to create lists of possible hyperparameters, and then the script will run experiments with all possible combinations.