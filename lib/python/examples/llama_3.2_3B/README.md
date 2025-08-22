# Federated-Learning Tensor Parallelism on Llama 3.2 3B

Only the GQA layers and FFN layers are split. The attention weights are split headwise. The FFN weights are split by hidden dimensions.

Attention weights make up 33.5% of total parameters
Feed-forward weights make up 50.2% of total parameters
Combined, they represent 83.7% of all model parameters
Embedding layers are difficult to split in our case, and norm layers are negligible. 

## Model Architecture
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/llama32.webp" width="700px">

  The Llama 3.2 3B model has been adapted for federated learning tensor parallelism by:
  
  1. **Removing Fairscale Dependencies**: All fairscale tensor parallel layers have been replaced with standard PyTorch layers
  2. **Implementing Tensor Parallelism**: Weight splitting and concatenation logic for federated learning with Flame

## Model Parameters

```json
{
  "dim": 3072,
  "ffn_dim_multiplier": 1.0,
  "multiple_of": 256,
  "n_heads": 24,
  "n_kv_heads": 8,
  "n_layers": 28,
  "norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "use_scaled_rope": true,
  "vocab_size": 128256,
  "max_seq_len": 128256,
}
```

## Tensor Parallelism Strategy

### Attention Layers (Split by Heads)
- **wq (Query)**: Split output dimension headwise across trainers
  - Shape: [3072, 24 * 128] → Split into [3072, heads_per_trainer * 128]
- **wk, wv (Key, Value)**: Split output dimension by KV heads across trainers  
  - Shape: [3072, 8 * 128] → Split into [3072, kv_heads_per_trainer * 128]
- **wo (Output)**: Split input dimension headwise across trainers
  - Shape: [24 * 128, 3072] → Split into [heads_per_trainer * 128, 3072]

### Feed-Forward Network (Split by Hidden Dimensions)
- **w1, w3**: Split output dimension evenly across trainers
  - Shape: [3072, hidden_dim] → Split into [3072, hidden_dim/world_size]
- **w2**: Split input dimension evenly across trainers
  - Shape: [hidden_dim, 3072] → Split into [hidden_dim/world_size, 3072]

### Replicated Layers (Duplicate and Average)
- **Token Embeddings** (tok_embeddings): Full replication
- **RMSNorm** layers: Full replication  
- **Output Layer**: Full replication

## Training Specifications

### Loss Function
- **Cross-Entropy Loss** for language modeling
- **Ignore padding tokens** in loss calculation
- **Perplexity** as evaluation metric

### Optimizer
- **AdamW** (standard for LLM fine-tuning)
- **Learning Rate**: 1e-5 (configurable)
- **Weight Decay**: 0.01 (recommended)
- **Beta1**: 0.9, **Beta2**: 0.999

### Data Format for Fine-tuning

#### Supported Formats:
1. **JSONL Format** (`.jsonl`):
   ```json
   {"text": "Your training text here..."}
   {"text": "Another training example..."}
   ```

2. **Plain Text Format** (`.txt`):
   ```
   Your training text here...
   Another training example...
   ```

#### Data Processing:
- **Tokenization**: Using Llama 3.2 tokenizer (tiktoken-based)
- **Sequence Length**: Configurable (default: 512 tokens)
- **Padding**: Pad shorter sequences with pad/eos tokens
- **Truncation**: Truncate longer sequences to max length

### Hyperparameters

#### Recommended Settings:
- **Batch Size**: 4-8 (depends on GPU memory)
- **Learning Rate**: 1e-5 to 5e-5
- **Max Sequence Length**: 512-2048 tokens
- **Training Rounds**: 5-20 (depends on dataset size)
- **World Size**: 2-8 trainers (for tensor parallelism)

#### Configurable in config.json:
```json
{
  "hyperparameters": {
    "rounds": 5,
    "world_size": 2,
    "learning_rate": 1e-5,
    "batch_size": 4,
    "max_seq_len": 512,
    "max_batch_size": 8,
    "seed": 42
  }
}
```

## Pretrained Weight Loading

### Source
- **File**: `consolidated.00.pth` (Llama 3.2 3B checkpoint)
- **Format**: PyTorch state dict
- **Automatic Adaptation**: Fairscale → Standard PyTorch layer mapping

### Loading Process
1. Load checkpoint from `consolidated.00.pth`
2. Map fairscale parameter names to standard PyTorch names
3. Raise error for shape mismatches

## Checkpointing and Results

### Experiment Results
- **File**: `experiment_results_llama32.pkl`
- **Contains**: (round, loss, perplexity) tuples
- **Key**: (world_size, lr, enable_swapping, rounds, seed)

### Model Checkpoints
- **Directory**: `checkpoints/`
- **Format**: `llama32_3b_ws{world_size}_lr{lr}_round{round}.pth`
- **Contains**: model_state_dict, experiment_key, round, config

### Evaluation Results
- **File**: `eval_res_llama32_ws{world_size}_r{rounds}.txt`
- **Format**: Round-by-round loss and perplexity

## Memory Requirements

### Estimation for Full Model:
- **Model Parameters**: ~3B parameters × 4 bytes = ~12GB
- **Gradients**: ~12GB  
- **Optimizer States**: ~24GB (AdamW)
- **Total per Trainer**: ~48GB GPU memory

### With Tensor Parallelism (world_size=2):
- **Per Trainer**: ~24GB GPU memory
- **Aggregator**: ~12GB GPU memory (model only)

## Future Enhancements

1. **Dynamic Batching**: Variable sequence lengths
2. **Gradient Clipping**: Prevent exploding gradients
3. **Learning Rate Scheduling**: Cosine annealing, warmup
4. **Mixed Precision**: FP16/BF16 training for memory efficiency
5. **Data Pipeline**: More sophisticated data loading and preprocessing

## Model Layers and Shapes


### Our Modified Model
tok_embeddings.weight | Shape: torch.Size([128256, 3072])
layers.0.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.0.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.0.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.0.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.0.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.0.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.0.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.0.attention_norm.weight | Shape: torch.Size([3072])
layers.0.ffn_norm.weight | Shape: torch.Size([3072])
layers.1.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.1.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.1.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.1.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.1.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.1.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.1.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.1.attention_norm.weight | Shape: torch.Size([3072])
layers.1.ffn_norm.weight | Shape: torch.Size([3072])
layers.2.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.2.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.2.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.2.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.2.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.2.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.2.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.2.attention_norm.weight | Shape: torch.Size([3072])
layers.2.ffn_norm.weight | Shape: torch.Size([3072])
layers.3.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.3.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.3.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.3.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.3.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.3.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.3.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.3.attention_norm.weight | Shape: torch.Size([3072])
layers.3.ffn_norm.weight | Shape: torch.Size([3072])
layers.4.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.4.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.4.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.4.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.4.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.4.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.4.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.4.attention_norm.weight | Shape: torch.Size([3072])
layers.4.ffn_norm.weight | Shape: torch.Size([3072])
layers.5.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.5.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.5.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.5.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.5.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.5.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.5.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.5.attention_norm.weight | Shape: torch.Size([3072])
layers.5.ffn_norm.weight | Shape: torch.Size([3072])
layers.6.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.6.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.6.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.6.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.6.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.6.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.6.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.6.attention_norm.weight | Shape: torch.Size([3072])
layers.6.ffn_norm.weight | Shape: torch.Size([3072])
layers.7.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.7.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.7.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.7.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.7.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.7.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.7.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.7.attention_norm.weight | Shape: torch.Size([3072])
layers.7.ffn_norm.weight | Shape: torch.Size([3072])
layers.8.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.8.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.8.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.8.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.8.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.8.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.8.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.8.attention_norm.weight | Shape: torch.Size([3072])
layers.8.ffn_norm.weight | Shape: torch.Size([3072])
layers.9.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.9.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.9.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.9.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.9.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.9.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.9.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.9.attention_norm.weight | Shape: torch.Size([3072])
layers.9.ffn_norm.weight | Shape: torch.Size([3072])
layers.10.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.10.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.10.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.10.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.10.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.10.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.10.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.10.attention_norm.weight | Shape: torch.Size([3072])
layers.10.ffn_norm.weight | Shape: torch.Size([3072])
layers.11.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.11.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.11.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.11.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.11.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.11.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.11.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.11.attention_norm.weight | Shape: torch.Size([3072])
layers.11.ffn_norm.weight | Shape: torch.Size([3072])
layers.12.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.12.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.12.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.12.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.12.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.12.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.12.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.12.attention_norm.weight | Shape: torch.Size([3072])
layers.12.ffn_norm.weight | Shape: torch.Size([3072])
layers.13.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.13.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.13.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.13.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.13.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.13.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.13.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.13.attention_norm.weight | Shape: torch.Size([3072])
layers.13.ffn_norm.weight | Shape: torch.Size([3072])
layers.14.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.14.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.14.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.14.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.14.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.14.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.14.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.14.attention_norm.weight | Shape: torch.Size([3072])
layers.14.ffn_norm.weight | Shape: torch.Size([3072])
layers.15.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.15.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.15.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.15.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.15.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.15.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.15.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.15.attention_norm.weight | Shape: torch.Size([3072])
layers.15.ffn_norm.weight | Shape: torch.Size([3072])
layers.16.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.16.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.16.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.16.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.16.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.16.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.16.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.16.attention_norm.weight | Shape: torch.Size([3072])
layers.16.ffn_norm.weight | Shape: torch.Size([3072])
layers.17.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.17.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.17.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.17.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.17.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.17.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.17.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.17.attention_norm.weight | Shape: torch.Size([3072])
layers.17.ffn_norm.weight | Shape: torch.Size([3072])
layers.18.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.18.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.18.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.18.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.18.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.18.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.18.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.18.attention_norm.weight | Shape: torch.Size([3072])
layers.18.ffn_norm.weight | Shape: torch.Size([3072])
layers.19.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.19.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.19.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.19.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.19.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.19.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.19.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.19.attention_norm.weight | Shape: torch.Size([3072])
layers.19.ffn_norm.weight | Shape: torch.Size([3072])
layers.20.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.20.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.20.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.20.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.20.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.20.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.20.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.20.attention_norm.weight | Shape: torch.Size([3072])
layers.20.ffn_norm.weight | Shape: torch.Size([3072])
layers.21.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.21.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.21.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.21.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.21.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.21.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.21.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.21.attention_norm.weight | Shape: torch.Size([3072])
layers.21.ffn_norm.weight | Shape: torch.Size([3072])
layers.22.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.22.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.22.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.22.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.22.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.22.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.22.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.22.attention_norm.weight | Shape: torch.Size([3072])
layers.22.ffn_norm.weight | Shape: torch.Size([3072])
layers.23.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.23.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.23.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.23.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.23.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.23.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.23.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.23.attention_norm.weight | Shape: torch.Size([3072])
layers.23.ffn_norm.weight | Shape: torch.Size([3072])
layers.24.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.24.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.24.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.24.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.24.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.24.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.24.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.24.attention_norm.weight | Shape: torch.Size([3072])
layers.24.ffn_norm.weight | Shape: torch.Size([3072])
layers.25.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.25.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.25.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.25.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.25.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.25.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.25.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.25.attention_norm.weight | Shape: torch.Size([3072])
layers.25.ffn_norm.weight | Shape: torch.Size([3072])
layers.26.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.26.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.26.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.26.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.26.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.26.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.26.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.26.attention_norm.weight | Shape: torch.Size([3072])
layers.26.ffn_norm.weight | Shape: torch.Size([3072])
layers.27.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.27.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.27.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.27.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.27.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.27.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.27.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.27.attention_norm.weight | Shape: torch.Size([3072])
layers.27.ffn_norm.weight | Shape: torch.Size([3072])
norm.weight | Shape: torch.Size([3072])
output.weight | Shape: torch.Size([128256, 3072])



### Llama3.2-3B from Meta AI
tok_embeddings.weight | Shape: torch.Size([128256, 3072])
layers.0.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.0.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.0.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.0.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.0.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.0.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.0.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.0.attention_norm.weight | Shape: torch.Size([3072])
layers.0.ffn_norm.weight | Shape: torch.Size([3072])
layers.1.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.1.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.1.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.1.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.1.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.1.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.1.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.1.attention_norm.weight | Shape: torch.Size([3072])
layers.1.ffn_norm.weight | Shape: torch.Size([3072])
layers.2.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.2.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.2.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.2.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.2.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.2.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.2.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.2.attention_norm.weight | Shape: torch.Size([3072])
layers.2.ffn_norm.weight | Shape: torch.Size([3072])
layers.3.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.3.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.3.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.3.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.3.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.3.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.3.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.3.attention_norm.weight | Shape: torch.Size([3072])
layers.3.ffn_norm.weight | Shape: torch.Size([3072])
layers.4.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.4.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.4.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.4.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.4.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.4.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.4.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.4.attention_norm.weight | Shape: torch.Size([3072])
layers.4.ffn_norm.weight | Shape: torch.Size([3072])
layers.5.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.5.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.5.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.5.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.5.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.5.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.5.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.5.attention_norm.weight | Shape: torch.Size([3072])
layers.5.ffn_norm.weight | Shape: torch.Size([3072])
layers.6.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.6.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.6.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.6.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.6.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.6.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.6.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.6.attention_norm.weight | Shape: torch.Size([3072])
layers.6.ffn_norm.weight | Shape: torch.Size([3072])
layers.7.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.7.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.7.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.7.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.7.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.7.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.7.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.7.attention_norm.weight | Shape: torch.Size([3072])
layers.7.ffn_norm.weight | Shape: torch.Size([3072])
layers.8.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.8.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.8.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.8.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.8.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.8.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.8.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.8.attention_norm.weight | Shape: torch.Size([3072])
layers.8.ffn_norm.weight | Shape: torch.Size([3072])
layers.9.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.9.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.9.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.9.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.9.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.9.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.9.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.9.attention_norm.weight | Shape: torch.Size([3072])
layers.9.ffn_norm.weight | Shape: torch.Size([3072])
layers.10.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.10.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.10.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.10.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.10.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.10.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.10.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.10.attention_norm.weight | Shape: torch.Size([3072])
layers.10.ffn_norm.weight | Shape: torch.Size([3072])
layers.11.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.11.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.11.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.11.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.11.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.11.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.11.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.11.attention_norm.weight | Shape: torch.Size([3072])
layers.11.ffn_norm.weight | Shape: torch.Size([3072])
layers.12.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.12.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.12.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.12.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.12.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.12.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.12.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.12.attention_norm.weight | Shape: torch.Size([3072])
layers.12.ffn_norm.weight | Shape: torch.Size([3072])
layers.13.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.13.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.13.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.13.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.13.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.13.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.13.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.13.attention_norm.weight | Shape: torch.Size([3072])
layers.13.ffn_norm.weight | Shape: torch.Size([3072])
layers.14.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.14.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.14.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.14.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.14.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.14.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.14.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.14.attention_norm.weight | Shape: torch.Size([3072])
layers.14.ffn_norm.weight | Shape: torch.Size([3072])
layers.15.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.15.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.15.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.15.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.15.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.15.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.15.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.15.attention_norm.weight | Shape: torch.Size([3072])
layers.15.ffn_norm.weight | Shape: torch.Size([3072])
layers.16.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.16.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.16.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.16.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.16.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.16.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.16.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.16.attention_norm.weight | Shape: torch.Size([3072])
layers.16.ffn_norm.weight | Shape: torch.Size([3072])
layers.17.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.17.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.17.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.17.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.17.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.17.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.17.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.17.attention_norm.weight | Shape: torch.Size([3072])
layers.17.ffn_norm.weight | Shape: torch.Size([3072])
layers.18.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.18.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.18.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.18.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.18.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.18.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.18.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.18.attention_norm.weight | Shape: torch.Size([3072])
layers.18.ffn_norm.weight | Shape: torch.Size([3072])
layers.19.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.19.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.19.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.19.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.19.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.19.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.19.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.19.attention_norm.weight | Shape: torch.Size([3072])
layers.19.ffn_norm.weight | Shape: torch.Size([3072])
layers.20.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.20.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.20.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.20.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.20.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.20.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.20.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.20.attention_norm.weight | Shape: torch.Size([3072])
layers.20.ffn_norm.weight | Shape: torch.Size([3072])
layers.21.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.21.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.21.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.21.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.21.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.21.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.21.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.21.attention_norm.weight | Shape: torch.Size([3072])
layers.21.ffn_norm.weight | Shape: torch.Size([3072])
layers.22.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.22.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.22.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.22.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.22.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.22.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.22.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.22.attention_norm.weight | Shape: torch.Size([3072])
layers.22.ffn_norm.weight | Shape: torch.Size([3072])
layers.23.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.23.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.23.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.23.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.23.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.23.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.23.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.23.attention_norm.weight | Shape: torch.Size([3072])
layers.23.ffn_norm.weight | Shape: torch.Size([3072])
layers.24.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.24.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.24.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.24.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.24.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.24.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.24.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.24.attention_norm.weight | Shape: torch.Size([3072])
layers.24.ffn_norm.weight | Shape: torch.Size([3072])
layers.25.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.25.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.25.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.25.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.25.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.25.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.25.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.25.attention_norm.weight | Shape: torch.Size([3072])
layers.25.ffn_norm.weight | Shape: torch.Size([3072])
layers.26.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.26.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.26.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.26.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.26.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.26.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.26.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.26.attention_norm.weight | Shape: torch.Size([3072])
layers.26.ffn_norm.weight | Shape: torch.Size([3072])
layers.27.attention.wq.weight | Shape: torch.Size([3072, 3072])
layers.27.attention.wk.weight | Shape: torch.Size([1024, 3072])
layers.27.attention.wv.weight | Shape: torch.Size([1024, 3072])
layers.27.attention.wo.weight | Shape: torch.Size([3072, 3072])
layers.27.feed_forward.w1.weight | Shape: torch.Size([8192, 3072])
layers.27.feed_forward.w3.weight | Shape: torch.Size([8192, 3072])
layers.27.feed_forward.w2.weight | Shape: torch.Size([3072, 8192])
layers.27.attention_norm.weight | Shape: torch.Size([3072])
layers.27.ffn_norm.weight | Shape: torch.Size([3072])
norm.weight | Shape: torch.Size([3072])
output.weight | Shape: torch.Size([128256, 3072])