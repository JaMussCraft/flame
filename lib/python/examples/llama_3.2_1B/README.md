# Federated-Learning Tensor Parallelism on Llama 3.2 1B

This is essentially the same as the 3B model's experiment setup. Please refer to the README in 3B's experiment.

## Running test inferences on Checkpoints

Note: the script needs to be run in `llama_model` repo

```bash
#!/bin/bash
NGPUS=1
CHECKPOINT_DIR=/home/cc/flame/lib/python/examples/llama_3.2_1B/checkpoints/llama32_3b_ws2_lr1e-05_round2
PYTHONPATH=$(git rev-parse --show-toplevel):/home/cc/flame/lib/python \
  torchrun --nproc_per_node=$NGPUS \
  -m models.llama3.scripts.completion $CHECKPOINT_DIR \
  --world_size $NGPUS
```