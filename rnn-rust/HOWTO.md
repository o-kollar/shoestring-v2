# How to Use: Modern RNN with Vectorized Autodiff Engine (Rust)

A CPU-optimized character-level (or BPE-tokenized) language model written in pure Rust. It features a **GRU + RWKV hybrid architecture** with Mixture-of-Experts, SwiGLU MLPs, RMSNorm, and a custom vectorized autodiff engine — accelerated via SIMD auto-vectorization and Rayon parallelism.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building](#building)
3. [Quick Start](#quick-start)
4. [Configuration Reference](#configuration-reference)
5. [Training](#training)
6. [Text Generation](#text-generation)
7. [Checkpointing](#checkpointing)
8. [BPE Tokenizer](#bpe-tokenizer)
9. [Reinforcement Learning (RLVR)](#reinforcement-learning-rlvr)
10. [Architecture Overview](#architecture-overview)
11. [Performance Tips](#performance-tips)
12. [Examples](#examples)

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

Run with all defaults (trains on a built-in sample sentence for 500 epochs, then generates text):

```bash
cargo run --release
```

Train on custom text:

```bash
cargo run --release -- --trainingText="the quick brown fox jumps over the lazy dog"
```

---

## Configuration Reference

All options are passed as `--key=value` command-line arguments.

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
| `--epochs`       | `500`    | Number of training epochs                      |
| `--learningRate` | `0.001`  | AdamW learning rate                            |
| `--weightDecay`  | `0.01`   | AdamW weight decay                             |
| `--batchSize`    | `4`      | Sequences per mini-batch                       |
| `--seqLength`    | `25`     | Sequence length for BPTT                       |
| `--logEvery`     | `25`     | Print training stats every N epochs            |

### Generation

| Option          | Default | Description                         |
|-----------------|---------|-------------------------------------|
| `--temperature` | `0.8`   | Sampling temperature (lower = more deterministic) |
| `--genLength`   | `100`   | Number of tokens to generate        |

### Checkpointing

| Option              | Default            | Description                                      |
|---------------------|--------------------|--------------------------------------------------|
| `--savePath`        | `./checkpoint.bin` | Path to save the final checkpoint                |
| `--saveEvery`       | `0`                | Save intermediate checkpoint every N epochs (0 = off) |
| `--loadCheckpoint`  | *(empty)*          | Path to load a checkpoint from before training   |
| `--saveOnComplete`  | `true`             | Save checkpoint when training finishes           |

### BPE Tokenizer

| Option              | Default              | Description                                     |
|---------------------|----------------------|-------------------------------------------------|
| `--useBPE`          | `false`              | Use BPE tokenizer instead of character-level    |
| `--trainBPE`        | `false`              | Run BPE training mode only (no model training)  |
| `--bpeVocabSize`    | `512`                | Target vocabulary size for BPE                  |
| `--bpeSavePath`     | `./tokenizer.json`   | Where to save the trained BPE tokenizer         |
| `--bpeLoadPath`     | *(empty)*            | Load a pre-trained BPE tokenizer from this path |
| `--bpeTrainingFile` | *(empty)*            | Path to a text file for BPE training data       |
| `--bpeTrainingText` | *(empty)*            | Inline text for BPE training                    |
| `--bpeMinFrequency` | `2`                  | Minimum pair frequency for a BPE merge          |

### Reinforcement Learning (RLVR)

| Option            | Default  | Description                                    |
|-------------------|----------|------------------------------------------------|
| `--doRL`          | `false`  | Run RL fine-tuning after supervised training   |
| `--rlTask`        | `copy`   | RL task: `copy`, `reverse`, or `arithmetic`    |
| `--rlEpisodes`    | `100`    | Number of RL training episodes                 |
| `--rlLearningRate`| `0.0001` | Learning rate for the RL optimizer             |

---

## Training

### Basic Training

```bash
cargo run --release -- --trainingText="hello world" --epochs=300 --hiddenSize=64
```

The training loop:
1. Tokenizes the input text (character-level by default)
2. Creates mini-batches of fixed-length sequences
3. Runs forward pass → computes cross-entropy loss → backpropagates → updates via AdamW
4. Logs loss and gradient norms at regular intervals
5. Generates a sample at the end

### Resume from Checkpoint

```bash
cargo run --release -- --loadCheckpoint=./checkpoint.bin --epochs=1000
```

Training resumes from the saved epoch and optimizer state.

### Multi-Layer / Larger Models

```bash
cargo run --release -- \
  --hiddenSize=128 \
  --numLayers=3 \
  --embedSize=64 \
  --numHeads=8 \
  --numExperts=4 \
  --topK=2 \
  --epochs=1000 \
  --learningRate=0.0005
```

---

## Text Generation

Text generation happens automatically after training completes. It uses **temperature-scaled softmax sampling** with the model's hidden state carried across tokens.

Control generation behavior:

```bash
cargo run --release -- --temperature=0.5 --genLength=200
```

- **Lower temperature** (e.g., `0.3`) → more deterministic, repetitive output
- **Higher temperature** (e.g., `1.2`) → more creative, potentially noisy output

---

## Checkpointing

Checkpoints are saved in a compact **bincode binary format** containing:
- All model weights
- AdamW optimizer momentum (`m`) and variance (`v`) states
- Model architecture config (hidden size, layers, etc.)
- Metadata (epoch, loss, timestamp)

### Save Periodically During Training

```bash
cargo run --release -- --saveEvery=50 --savePath=./model.bin
```

This creates `model_epoch50.bin`, `model_epoch100.bin`, etc., plus a final `model.bin`.

### Load and Continue

```bash
cargo run --release -- --loadCheckpoint=./model.bin --epochs=2000
```

---

## BPE Tokenizer

The built-in BPE (Byte Pair Encoding) tokenizer can compress text into sub-word tokens for more efficient training on larger corpora.

### Step 1: Train a BPE Tokenizer

```bash
cargo run --release -- --trainBPE=true --bpeVocabSize=512 --bpeTrainingFile=./corpus.txt --bpeSavePath=./tokenizer.json
```

This outputs a `tokenizer.json` file with the learned merge rules and vocabulary.

### Step 2: Train the Model with BPE

```bash
cargo run --release -- --useBPE=true --bpeLoadPath=./tokenizer.json --trainingText="your training text here" --epochs=500
```

### Train BPE from Inline Text

```bash
cargo run --release -- --trainBPE=true --bpeTrainingText="your training corpus here" --bpeVocabSize=256
```

---

## Reinforcement Learning (RLVR)

After supervised pre-training, you can fine-tune the model with **GRPO** (Group Relative Policy Optimization) on verifiable tasks.

### Available Tasks

| Task         | Prompt Format | Expected Output  | Example              |
|--------------|---------------|------------------|----------------------|
| `copy`       | `abc:`        | `abc`            | `hello:` → `hello`  |
| `reverse`    | `abc>`        | `cba`            | `hello>` → `olleh`  |
| `arithmetic` | `3+5=`        | `8`              | `3*4=` → `12`       |

### Run RL Fine-Tuning

```bash
cargo run --release -- \
  --epochs=500 \
  --doRL=true \
  --rlTask=copy \
  --rlEpisodes=200 \
  --rlLearningRate=0.0001
```

The RL phase runs after supervised training completes, then generates a post-RL sample.

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
   cargo run --release -- --numExperts=8
   ```

---

## Examples

### Minimal Character-Level Training

```bash
cargo run --release -- --trainingText="abcabcabc" --hiddenSize=16 --epochs=200
```

### Full-Featured Training Pipeline

```bash
# 1. Train a BPE tokenizer on a corpus file
cargo run --release -- --trainBPE=true --bpeTrainingFile=./data.txt --bpeVocabSize=1024

# 2. Train the model using BPE tokens
cargo run --release -- \
  --useBPE=true \
  --bpeLoadPath=./tokenizer.json \
  --trainingText="$(cat ./data.txt)" \
  --hiddenSize=128 \
  --numLayers=2 \
  --embedSize=64 \
  --epochs=1000 \
  --saveEvery=100 \
  --savePath=./model.bin

# 3. Fine-tune with RL on arithmetic
cargo run --release -- \
  --loadCheckpoint=./model.bin \
  --epochs=0 \
  --doRL=true \
  --rlTask=arithmetic \
  --rlEpisodes=500
```

### Quick Smoke Test

```bash
cargo run --release -- --hiddenSize=16 --epochs=50 --fastMode=true --logEvery=10
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

Temperature: 0.8
Generated (100 tokens):
----------------------------------------------------------------------
hello world this is a modern rnn with rmsnorm residual connections...
----------------------------------------------------------------------
```

---

## License

See the project root for license information.
