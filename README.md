# A CPU-optimized language model written in pure Rust. 
It features a Mixture-of-Experts **GRU + RWKV hybrid architecture**.
---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building](#building)
3. [Quick Start](#quick-start)
4. [Commands](#commands)
5. [Configuration Reference](#configuration-reference)
6. [Train Tokenizer](#train-tokenizer)
7. [Train Model](#train-model)
8. [Inference](#inference)
9. [RLVR (Reinforcement Learning)](#rlvr-reinforcement-learning)
10. [Checkpointing](#checkpointing)
11. [Architecture Overview](#architecture-overview)
12. [Performance Tips](#performance-tips)
13. [Full Pipeline Example](#full-pipeline-example)

---

## Prerequisites

- **Rust toolchain** (1.70+ recommended) — install via [rustup](https://rustup.rs/)
- No GPU required — the engine is fully CPU-based

### Dependencies (managed by Cargo)

| Crate              | Purpose                                |
|--------------------|----------------------------------------|
| `serde` / `serde_json` | Serialization for checkpoints & BPE |
| `bincode`          | Binary checkpoint format               |
| `rand`             | Random number generation               |
| `rayon`            | Data-parallel expert forward passes    |
| `matrixmultiply`   | BLAS-quality matrix multiplication     |

---

## Building

```bash
# Navigate to the project directory
cd rnn-rust

# Debug build (fast compile, slow execution)
cargo build

# Release build (slow compile, fast execution — recommended)
cargo build --release
```

The release profile is configured with `opt-level = 3`, LTO, and `codegen-units = 1` for maximum performance and SIMD auto-vectorization.

---

## Quick Start

The CLI uses **subcommands** to separate each stage of the pipeline. Running with no command prints help:

```bash
cargo run --release
```

### Minimal end-to-end example

```bash
# 1. Train a model on inline text
cargo run --release -- train --trainingText="the quick brown fox jumps over the lazy dog" --epochs=300

# 2. Generate text from the trained checkpoint
cargo run --release -- inference --loadCheckpoint=./checkpoint.bin --trainingText="the quick brown fox jumps over the lazy dog"
```

---

## Commands

| Command            | Purpose                                         |
|--------------------|------------------------------------------------|
| `train-tokenizer`  | Train a BPE tokenizer on text data              |
| `train`            | Train the RNN model (with optional BPE)         |
| `inference`        | Generate text from a pre-trained model           |
| `rlvr`             | Run RLVR (GRPO) on a pre-trained model           |

Each command accepts `--key=value` options. The first positional argument (without `--`) is the command.

```bash
cargo run --release -- <command> [--option=value ...]
```

Aliases: `inference` can also be written as `generate` or `infer`. `rlvr` can also be written as `rl`.

---

## Configuration Reference

All options are passed as `--key=value` command-line arguments after the command name.

### Model Architecture

| Option         | Default | Description                                      |
|----------------|---------|--------------------------------------------------|
| `--hiddenSize` | `32`    | Hidden state dimension                           |
| `--numLayers`  | `1`     | Number of stacked RNN layers                     |
| `--embedSize`  | `16`    | Token embedding dimension                        |
| `--numHeads`   | `4`     | Number of attention heads (RWKV)                 |
| `--numExperts` | `4`     | Number of MoE experts per layer                  |
| `--topK`       | `1`     | How many experts to route to (besides expert 0)  |
| `--fastMode`   | `false` | If `true`, disables MoE MLP layers (lighter model) |
| `--useRWKV`    | `true`  | Enable RWKV-style linear attention in GRU cells  |

### Training

| Option           | Default  | Description                                    |
|------------------|----------|------------------------------------------------|
| `--trainingText` | *(built-in sample)* | The text corpus to train on          |
| `--trainingFile` | *(empty)* | Path to a text file for training data         |
| `--epochs`       | `500`    | Number of training epochs                      |
| `--learningRate` | `0.001`  | AdamW learning rate                            |
| `--weightDecay`  | `0.01`   | AdamW weight decay                             |
| `--batchSize`    | `4`      | Sequences per mini-batch                       |
| `--seqLength`    | `180`    | Sequence length for BPTT                       |
| `--logEvery`     | `25`     | Print training stats every N epochs            |

### Inference / Generation

| Option          | Default | Description                         |
|-----------------|---------|-------------------------------------|
| `--temperature` | `0.3`   | Sampling temperature (lower = more deterministic) |
| `--genLength`   | `100`   | Number of tokens to generate        |
| `--prompt`      | *(empty)* | Seed text for generation (inference mode) |

### Checkpointing

| Option              | Default            | Description                                      |
|---------------------|--------------------|--------------------------------------------------|
| `--savePath`        | `./checkpoint.bin` | Path to save the final checkpoint                |
| `--saveEvery`       | `0`                | Save intermediate checkpoint every N epochs (0 = off) |
| `--loadCheckpoint`  | *(empty)*          | Path to load a checkpoint from (required for `inference` and `rlvr`) |
| `--saveOnComplete`  | `true`             | Save checkpoint when training/RLVR finishes      |

### BPE Tokenizer

| Option              | Default              | Description                                     |
|---------------------|----------------------|-------------------------------------------------|
| `--useBPE`          | `false`              | Use BPE tokenizer instead of character-level    |
| `--bpeVocabSize`    | `512`                | Target vocabulary size for BPE                  |
| `--bpeSavePath`     | `./tokenizer.json`   | Where to save the trained BPE tokenizer         |
| `--bpeLoadPath`     | *(empty)*            | Load a pre-trained BPE tokenizer from this path |
| `--bpeTrainingFile` | *(empty)*            | Path to a text file for BPE training data       |
| `--bpeTrainingText` | *(empty)*            | Inline text for BPE training                    |
| `--bpeMinFrequency` | `2`                  | Minimum pair frequency for a BPE merge          |

### Reinforcement Learning (RLVR)

| Option            | Default  | Description                                    |
|-------------------|----------|------------------------------------------------|
| `--rlTask`        | `copy`   | RL task: `copy`, `reverse`, or `arithmetic`    |
| `--rlEpisodes`    | `100`    | Number of RL training episodes                 |
| `--rlLearningRate`| `0.0001` | Learning rate for the RL optimizer             |

---

## Train Tokenizer

The `train-tokenizer` command trains a BPE (Byte Pair Encoding) tokenizer independently of the model. This produces a `tokenizer.json` file that can later be used during model training and inference.

### From a file

```bash
cargo run --release -- train-tokenizer \
  --bpeTrainingFile=./corpus.txt \
  --bpeVocabSize=512 \
  --bpeSavePath=./tokenizer.json
```

### From inline text

```bash
cargo run --release -- train-tokenizer \
  --bpeTrainingText="your training corpus here" \
  --bpeVocabSize=256
```

### Key options

- **`--bpeVocabSize`** — Controls how many merge operations to learn (larger = more compression)
- **`--bpeMinFrequency`** — Pairs occurring less than this many times won't be merged
- **`--bpeSavePath`** — Output path for the tokenizer JSON file (default: `./tokenizer.json`)

The tokenizer training prioritizes data sources in this order: `--bpeTrainingFile` > `--bpeTrainingText` > `--trainingFile` > `--trainingText`.

---

## Train Model

The `train` command trains the RNN on text data. It supports character-level tokenization (default) or BPE tokenization with a pre-trained tokenizer. Training can start from scratch or resume from a checkpoint.

### Character-level training

```bash
cargo run --release -- train --trainingText="hello world" --epochs=300 --hiddenSize=64
```

### Training from a file

```bash
cargo run --release -- train --trainingFile=./data.txt --epochs=500
```

### Training with a pre-trained BPE tokenizer

```bash
cargo run --release -- train \
  --useBPE=true \
  --bpeLoadPath=./tokenizer.json \
  --trainingFile=./data.txt \
  --epochs=500
```

### Resume training from a checkpoint

```bash
cargo run --release -- train --loadCheckpoint=./checkpoint.bin --epochs=1000
```

Training resumes from the saved epoch and optimizer state. The model architecture is restored from the checkpoint.

### Multi-layer / larger models

```bash
cargo run --release -- train \
  --hiddenSize=128 \
  --numLayers=3 \
  --embedSize=64 \
  --numHeads=8 \
  --numExperts=4 \
  --topK=2 \
  --epochs=1000 \
  --learningRate=0.0005 \
  --trainingFile=./data.txt
```

### What happens during training

1. Tokenizes the input text (character-level or BPE)
2. Creates mini-batches of fixed-length sequences
3. Runs forward pass → computes cross-entropy loss → backpropagates → updates via AdamW
4. Logs loss and gradient norms at regular intervals
5. Saves checkpoint on completion (and optionally at intervals via `--saveEvery`)
6. Generates a sample at the end

---

## Inference

The `inference` command generates text from a pre-trained model checkpoint **without any training**. It requires `--loadCheckpoint`.

### Basic generation

```bash
cargo run --release -- inference \
  --loadCheckpoint=./checkpoint.bin \
  --trainingText="the quick brown fox jumps over the lazy dog"
```

> **Note:** You must provide the same training data source (via `--trainingText`, `--trainingFile`, or `--useBPE` + `--bpeLoadPath`) so the vocabulary mapping is reconstructed correctly.

### Generation with a prompt

Use `--prompt` to seed the model with specific text before generating:

```bash
cargo run --release -- inference \
  --loadCheckpoint=./checkpoint.bin \
  --trainingFile=./data.txt \
  --prompt="hello" \
  --temperature=0.5 \
  --genLength=200
```

The prompt is fed through the model token-by-token to build up the hidden state, then generation continues from that context.

### With BPE tokenizer

```bash
cargo run --release -- inference \
  --loadCheckpoint=./checkpoint.bin \
  --useBPE=true \
  --bpeLoadPath=./tokenizer.json \
  --trainingFile=./data.txt \
  --genLength=200
```

### Generation parameters

- **`--temperature`** — Controls randomness. Lower (e.g., `0.3`) = more deterministic; higher (e.g., `1.2`) = more creative
- **`--genLength`** — Number of tokens to generate (default: 100)
- **`--prompt`** — Optional seed text to condition generation on

---

## RLVR (Reinforcement Learning)

The `rlvr` command runs **GRPO** (Group Relative Policy Optimization) on a pre-trained model checkpoint. This fine-tunes the model on verifiable tasks without needing additional supervised data.

### Available Tasks

| Task         | Prompt Format | Expected Output  | Example              |
|--------------|---------------|------------------|----------------------|
| `copy`       | `abc:`        | `abc`            | `hello:` → `hello`  |
| `reverse`    | `abc>`        | `cba`            | `hello>` → `olleh`  |
| `arithmetic` | `3+5=`        | `8`              | `3*4=` → `12`       |

### Run RLVR on a pre-trained model

```bash
cargo run --release -- rlvr \
  --loadCheckpoint=./checkpoint.bin \
  --trainingText="the quick brown fox jumps over the lazy dog" \
  --rlTask=copy \
  --rlEpisodes=200 \
  --rlLearningRate=0.0001
```

### Arithmetic fine-tuning

```bash
cargo run --release -- rlvr \
  --loadCheckpoint=./checkpoint.bin \
  --trainingText="0123456789+-*=" \
  --rlTask=arithmetic \
  --rlEpisodes=500
```

### What happens during RLVR

1. Loads model weights from the checkpoint
2. For each episode, generates a random prompt for the task
3. Samples multiple completions and scores them with a verifiable reward function
4. Computes advantages via z-scored rewards (GRPO)
5. Updates the policy with a single gradient step per episode
6. Saves the fine-tuned model on completion

---

## Checkpointing

Checkpoints are saved in a compact **bincode binary format** containing:
- All model weights
- AdamW optimizer momentum (`m`) and variance (`v`) states
- Model architecture config (hidden size, layers, etc.)
- Metadata (epoch, loss, timestamp)

### Save periodically during training

```bash
cargo run --release -- train --saveEvery=50 --savePath=./model.bin --trainingFile=./data.txt
```

This creates `model_epoch50.bin`, `model_epoch100.bin`, etc., plus a final `model.bin`.

### Load and continue training

```bash
cargo run --release -- train --loadCheckpoint=./model.bin --epochs=2000
```

### Use a checkpoint for inference

```bash
cargo run --release -- inference --loadCheckpoint=./model.bin --trainingFile=./data.txt
```

---

## Architecture Overview

```
Input Token
    │
    ▼
┌──────────┐
│ Embedding │  (vocab_size × embed_size)
└──────────┘
    │
    ▼
┌───────────────────────────────────┐
│  ModernRNNLayer (× num_layers)    │
│                                   │
│  ┌─────────────────────────────┐  │
│  │ ModernGRUCell               │  │
│  │  • RMSNorm on input/hidden  │  │
│  │  • GRU gates (z, r, h_cand) │  │
│  │  • RWKV linear attention    │  │
│  │    (token-shift mixing,     │  │
│  │     dynamic decay, WKV)     │  │
│  └─────────────────────────────┘  │
│           │                       │
│           ▼                       │
│  ┌─────────────────────────────┐  │
│  │ Mixture of Experts (MoE)    │  │
│  │  • RMSNorm                  │  │
│  │  • Router (softmax top-k)   │  │
│  │  • SwiGLU Expert FFNs       │  │
│  │  • Always-on expert 0       │  │
│  │  • Rayon-parallel forward   │  │
│  └─────────────────────────────┘  │
│           │                       │
│     + Residual Connection         │
└───────────────────────────────────┘
    │
    ▼
┌──────────┐
│ RMSNorm  │
└──────────┘
    │
    ▼
┌──────────────┐
│ Output Proj  │  (hidden_size × vocab_size)
└──────────────┘
    │
    ▼
  Logits → Softmax → Sample / Cross-Entropy Loss
```

### Key Components

- **Vectorized Autodiff Engine** — A custom tape-based automatic differentiation engine operating on dense tensors (no per-scalar graph nodes)
- **GRU Cell** — Gated Recurrent Unit with RMSNorm-normalized inputs and hidden states
- **RWKV Attention** — Linear-complexity attention with token-shift mixing, dynamic time-decay, and WKV (weighted key-value) state
- **Mixture of Experts** — Sparse routing with a permanent "expert 0" and top-k soft-routed additional experts
- **SwiGLU** — Gated linear unit with SiLU activation for expert FFN layers
- **AdamW Optimizer** — With bias-corrected moments and decoupled weight decay

---

## Performance Tips

1. **Always use `--release`** — The release profile enables `opt-level=3`, LTO, and single codegen unit for maximum SIMD auto-vectorization
2. **Use `--fastMode=true`** for quick experiments — disables MoE MLP layers, significantly reducing compute
3. **Reduce `--numExperts`** if training is slow — each expert adds parameters and compute
4. **Increase `--batchSize`** for smoother gradients (at the cost of memory)
5. **Lower `--seqLength`** to reduce memory usage and speed up each step
6. **Set `RAYON_NUM_THREADS`** environment variable to control parallelism:
   ```bash
   set RAYON_NUM_THREADS=4
   cargo run --release -- train --numExperts=8
   ```

---

## Full Pipeline Example

A complete workflow from tokenizer training through RL fine-tuning:

```bash
# 1. Train a BPE tokenizer on a corpus file
cargo run --release -- train-tokenizer \
  --bpeTrainingFile=./data.txt \
  --bpeVocabSize=1024 \
  --bpeSavePath=./tokenizer.json

# 2. Train the model using BPE tokens
cargo run --release -- train \
  --useBPE=true \
  --bpeLoadPath=./tokenizer.json \
  --trainingFile=./data.txt \
  --hiddenSize=128 \
  --numLayers=2 \
  --embedSize=64 \
  --epochs=1000 \
  --saveEvery=100 \
  --savePath=./model.bin

# 3. Generate text from the trained model
cargo run --release -- inference \
  --loadCheckpoint=./model.bin \
  --useBPE=true \
  --bpeLoadPath=./tokenizer.json \
  --trainingFile=./data.txt \
  --prompt="Once upon" \
  --temperature=0.5 \
  --genLength=200

# 4. Fine-tune with RL on arithmetic
cargo run --release -- rlvr \
  --loadCheckpoint=./model.bin \
  --useBPE=true \
  --bpeLoadPath=./tokenizer.json \
  --trainingFile=./data.txt \
  --rlTask=arithmetic \
  --rlEpisodes=500
```

### Quick Smoke Test

```bash
cargo run --release -- train --hiddenSize=16 --epochs=50 --fastMode=true --logEvery=10
```

---

## Output Format

During training you will see output like:

```
Epoch   25 | Loss: 2.845123 | ∇: 0.4521 | Time: 1.2s
Epoch   50 | Loss: 2.103456 | ∇: 0.3210 | Time: 2.4s
...
Done in 45.3s | Best loss: 0.234567

Saved binary checkpoint to ./checkpoint.bin (0.15 MB)

Generating sample...

Temperature: 0.3
Generated (100 tokens):
----------------------------------------------------------------------
hello world this is a modern rnn with rmsnorm residual connections...
----------------------------------------------------------------------
```

---

## Legacy Flag Support

For backward compatibility, the old flag-based usage still works when no subcommand is provided:

- `--trainBPE=true` → equivalent to the `train-tokenizer` command
- `--doRL=true` → runs training then RL (use separate `train` + `rlvr` commands instead)

These are deprecated; prefer the subcommand-based interface.

---



