// ============================================================================
// MODERN RNN WITH VECTORIZED AUTODIFF ENGINE - Rust Version (2026)
// CPU-optimized with SIMD + Rayon parallelism — No scalar graph nodes
// ============================================================================

use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Clone)]
struct Config {
    hidden_size: usize,
    num_layers: usize,
    embed_size: usize,
    seq_length: usize,
    learning_rate: f32,
    weight_decay: f32,
    epochs: usize,
    batch_size: usize,
    num_experts: usize,
    top_k: usize,
    num_heads: usize,
    fast_mode: bool,
    use_rwkv: bool,
    temperature: f32,
    gen_length: usize,
    log_every: usize,
    training_text: String,
    training_file: String,
    rl_episodes: usize,
    rl_task: String,
    rl_learning_rate: f32,
    do_rl: bool,
    save_every: usize,
    save_path: String,
    load_checkpoint: String,
    save_on_complete: bool,
    use_bpe: bool,
    train_bpe: bool,
    bpe_vocab_size: usize,
    bpe_save_path: String,
    bpe_load_path: String,
    bpe_training_file: String,
    bpe_training_text: String,
    bpe_min_frequency: usize,
}

impl Config {
    fn default_config() -> Self {
        Config {
            hidden_size: 32,
            num_layers: 1,
            embed_size: 16,
            seq_length: 180,
            learning_rate: 0.001,
            weight_decay: 0.01,
            epochs: 500,
            batch_size: 4,
            num_experts: 4,
            top_k: 1,
            num_heads: 4,
            fast_mode: false,
            use_rwkv: true,
            temperature: 0.3,
            gen_length: 100,
            log_every: 25,
            training_text: "hello world this is a modern rnn with rmsnorm residual connections gru gates and swiglu mlp learning to predict characters in vanilla javascript".to_string(),
            training_file: String::new(),
            rl_episodes: 100,
            rl_task: "copy".to_string(),
            rl_learning_rate: 0.0001,
            do_rl: false,
            save_every: 0,
            save_path: "./checkpoint.bin".to_string(),
            load_checkpoint: String::new(),
            save_on_complete: true,
            use_bpe: false,
            train_bpe: false,
            bpe_vocab_size: 512,
            bpe_save_path: "./tokenizer.json".to_string(),
            bpe_load_path: String::new(),
            bpe_training_file: String::new(),
            bpe_training_text: String::new(),
            bpe_min_frequency: 2,
        }
    }

    fn from_args() -> Self {
        let mut config = Self::default_config();
        for arg in env::args().skip(1) {
            let arg = arg.trim_start_matches("--");
            if let Some((key, value)) = arg.split_once('=') {
                match key.to_lowercase().as_str() {
                    "hiddensize" => config.hidden_size = value.parse().unwrap_or(config.hidden_size),
                    "numlayers" => config.num_layers = value.parse().unwrap_or(config.num_layers),
                    "embedsize" => config.embed_size = value.parse().unwrap_or(config.embed_size),
                    "seqlength" => config.seq_length = value.parse().unwrap_or(config.seq_length),
                    "learningrate" => config.learning_rate = value.parse().unwrap_or(config.learning_rate),
                    "weightdecay" => config.weight_decay = value.parse().unwrap_or(config.weight_decay),
                    "epochs" => config.epochs = value.parse().unwrap_or(config.epochs),
                    "batchsize" => config.batch_size = value.parse().unwrap_or(config.batch_size),
                    "numexperts" => config.num_experts = value.parse().unwrap_or(config.num_experts),
                    "topk" => config.top_k = value.parse().unwrap_or(config.top_k),
                    "numheads" => config.num_heads = value.parse().unwrap_or(config.num_heads),
                    "fastmode" => config.fast_mode = value == "true",
                    "userwkv" => config.use_rwkv = value == "true",
                    "temperature" => config.temperature = value.parse().unwrap_or(config.temperature),
                    "genlength" => config.gen_length = value.parse().unwrap_or(config.gen_length),
                    "logevery" => config.log_every = value.parse().unwrap_or(config.log_every),
                    "trainingtext" => config.training_text = value.to_string(),
                    "trainingfile" => config.training_file = value.to_string(),
                    "rlepisodes" => config.rl_episodes = value.parse().unwrap_or(config.rl_episodes),
                    "rltask" => config.rl_task = value.to_string(),
                    "rllearningrate" => config.rl_learning_rate = value.parse().unwrap_or(config.rl_learning_rate),
                    "dorl" => config.do_rl = value == "true",
                    "saveevery" => config.save_every = value.parse().unwrap_or(config.save_every),
                    "savepath" => config.save_path = value.to_string(),
                    "loadcheckpoint" => config.load_checkpoint = value.to_string(),
                    "saveoncomplete" => config.save_on_complete = value == "true",
                    "usebpe" => config.use_bpe = value == "true",
                    "trainbpe" => config.train_bpe = value == "true",
                    "bpevocabsize" => config.bpe_vocab_size = value.parse().unwrap_or(config.bpe_vocab_size),
                    "bpesavepath" => config.bpe_save_path = value.to_string(),
                    "bpeloadpath" => config.bpe_load_path = value.to_string(),
                    "bpetrainingfile" => config.bpe_training_file = value.to_string(),
                    "bpetrainingtext" => config.bpe_training_text = value.to_string(),
                    "bpeminfrequency" => config.bpe_min_frequency = value.parse().unwrap_or(config.bpe_min_frequency),
                    _ => {}
                }
            }
        }
        config
    }
}

// ============================================================================
// SIMD-FRIENDLY VECTOR OPERATIONS
// These tight loops auto-vectorize with rustc -C opt-level=3 + LTO
// ============================================================================

#[inline]
fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[inline]
fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, y) in a.iter_mut().zip(b.iter()) { *x += y; }
}

#[inline]
fn vec_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

#[inline]
fn vec_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

#[inline]
fn vec_scale(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|x| x * s).collect()
}

#[inline]
fn vec_scale_inplace(a: &mut [f32], s: f32) {
    for x in a.iter_mut() { *x *= s; }
}

#[inline]
fn vec_sigmoid(a: &[f32]) -> Vec<f32> {
    a.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

#[inline]
fn vec_tanh(a: &[f32]) -> Vec<f32> {
    a.iter().map(|&x| x.tanh()).collect()
}

#[inline]
fn vec_silu(a: &[f32]) -> Vec<f32> {
    a.iter().map(|&x| { let s = 1.0 / (1.0 + (-x).exp()); x * s }).collect()
}

#[inline]
fn vec_exp(a: &[f32]) -> Vec<f32> {
    a.iter().map(|&x| x.exp()).collect()
}

#[inline]
fn vec_clamp(a: &[f32], lo: f32, hi: f32) -> Vec<f32> {
    a.iter().map(|&x| x.clamp(lo, hi)).collect()
}

#[inline]
fn vec_max(a: &[f32]) -> f32 {
    a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

// ============================================================================
// VECTORIZED TENSOR AUTODIFF ENGINE
// Each node = dense row-major matrix. No per-scalar nodes.
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TID(usize);

#[derive(Clone, Debug)]
enum TensorOp {
    None,
    MatMul { a: TID, b: TID, m: usize, k: usize, n: usize },
    Add { a: TID, b: TID },
    Mul { a: TID, b: TID },
    Sub { a: TID, b: TID },
    Scale { a: TID, s: f32 },
    AddScalar { a: TID, s: f32 },
    Sigmoid { a: TID },
    Tanh { a: TID },
    SiLU { a: TID },
    Exp { a: TID },
    Log { a: TID },
    Square { a: TID },
    Sqrt { a: TID },
    Neg { a: TID },
    Pow { a: TID, n: f32 },
    Clamp { a: TID, lo: f32, hi: f32 },
    BroadcastAdd { a: TID, bias: TID, rows: usize, cols: usize },
    BroadcastMul { a: TID, scale: TID, rows: usize, cols: usize },
    SoftmaxCE { logits: TID, target_idx: usize, vocab: usize },
    ReduceMeanCols { a: TID, rows: usize, cols: usize },
    ExpandCol { a: TID, rows: usize, cols: usize },
    Transpose { a: TID, rows: usize, cols: usize },
    DivElem { a: TID, b: TID },
    OneMinus { a: TID },
    RowSlice { a: TID, row: usize, cols: usize },
    ScalarDiv { a: TID, n: f32 },
    SumAll { a: TID },
}

struct TensorNode {
    data: Vec<f32>,
    grad: Vec<f32>,
    rows: usize,
    cols: usize,
    op: TensorOp,
    is_param: bool,
}

struct Graph {
    nodes: Vec<TensorNode>,
    param_boundary: usize,
}

impl Graph {
    fn new() -> Self {
        Graph { nodes: Vec::with_capacity(4096), param_boundary: 0 }
    }

    fn param(&mut self, data: Vec<f32>, rows: usize, cols: usize) -> TID {
        let len = rows * cols;
        debug_assert_eq!(data.len(), len);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows, cols,
            op: TensorOp::None, is_param: true,
        });
        TID(id)
    }

    fn constant(&mut self, data: Vec<f32>, rows: usize, cols: usize) -> TID {
        let len = rows * cols;
        debug_assert_eq!(data.len(), len);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows, cols,
            op: TensorOp::None, is_param: false,
        });
        TID(id)
    }

    fn freeze_params(&mut self) { self.param_boundary = self.nodes.len(); }

    fn reset(&mut self) { self.nodes.truncate(self.param_boundary); }

    fn zero_grad(&mut self) {
        for node in self.nodes.iter_mut() {
            for g in node.grad.iter_mut() { *g = 0.0; }
        }
    }

    fn data(&self, t: TID) -> &[f32] { &self.nodes[t.0].data }
    fn cols(&self, t: TID) -> usize { self.nodes[t.0].cols }

    // ------------------------------------------------------------------
    // Forward ops — each builds one TensorNode
    // ------------------------------------------------------------------

    fn matmul(&mut self, a: TID, b: TID) -> TID {
        let m = self.nodes[a.0].rows;
        let k = self.nodes[a.0].cols;
        let n = self.nodes[b.0].cols;
        debug_assert_eq!(k, self.nodes[b.0].rows, "matmul shape [{},{}] @ [{},{}]",
            m, k, self.nodes[b.0].rows, n);

        let a_data = &self.nodes[a.0].data;
        let b_data = &self.nodes[b.0].data;
        let mut out = vec![0.0f32; m * n];

        // BLAS-quality matmul via matrixmultiply crate
        unsafe {
            matrixmultiply::sgemm(
                m, k, n,
                1.0,
                a_data.as_ptr(), k as isize, 1,  // A: row-major [m, k]
                b_data.as_ptr(), n as isize, 1,   // B: row-major [k, n]
                0.0,
                out.as_mut_ptr(), n as isize, 1,  // C: row-major [m, n]
            );
        }

        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; m * n], rows: m, cols: n,
            op: TensorOp::MatMul { a, b, m, k, n }, is_param: false,
        });
        TID(id)
    }

    fn add(&mut self, a: TID, b: TID) -> TID {
        let data = vec_add(&self.nodes[a.0].data, &self.nodes[b.0].data);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Add { a, b }, is_param: false,
        });
        TID(id)
    }

    fn sub(&mut self, a: TID, b: TID) -> TID {
        let data = vec_sub(&self.nodes[a.0].data, &self.nodes[b.0].data);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Sub { a, b }, is_param: false,
        });
        TID(id)
    }

    fn mul(&mut self, a: TID, b: TID) -> TID {
        let data = vec_mul(&self.nodes[a.0].data, &self.nodes[b.0].data);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Mul { a, b }, is_param: false,
        });
        TID(id)
    }

    fn div_elem(&mut self, a: TID, b: TID) -> TID {
        let ad = &self.nodes[a.0].data;
        let bd = &self.nodes[b.0].data;
        let data: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x / (y + 1e-8)).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::DivElem { a, b }, is_param: false,
        });
        TID(id)
    }

    fn scale(&mut self, a: TID, s: f32) -> TID {
        let data = vec_scale(&self.nodes[a.0].data, s);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Scale { a, s }, is_param: false,
        });
        TID(id)
    }

    fn add_scalar(&mut self, a: TID, s: f32) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|x| x + s).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::AddScalar { a, s }, is_param: false,
        });
        TID(id)
    }

    fn scalar_div(&mut self, a: TID, n: f32) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|x| x / n).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::ScalarDiv { a, n }, is_param: false,
        });
        TID(id)
    }

    fn neg(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|x| -x).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Neg { a }, is_param: false,
        });
        TID(id)
    }

    fn sigmoid(&mut self, a: TID) -> TID {
        let data = vec_sigmoid(&self.nodes[a.0].data);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Sigmoid { a }, is_param: false,
        });
        TID(id)
    }

    fn tanh_op(&mut self, a: TID) -> TID {
        let data = vec_tanh(&self.nodes[a.0].data);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Tanh { a }, is_param: false,
        });
        TID(id)
    }

    fn silu(&mut self, a: TID) -> TID {
        let data = vec_silu(&self.nodes[a.0].data);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::SiLU { a }, is_param: false,
        });
        TID(id)
    }

    fn exp_op(&mut self, a: TID) -> TID {
        let data = vec_exp(&self.nodes[a.0].data);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Exp { a }, is_param: false,
        });
        TID(id)
    }

    fn log_op(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|&x| (x + 1e-8).ln()).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Log { a }, is_param: false,
        });
        TID(id)
    }

    fn square(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|x| x * x).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Square { a }, is_param: false,
        });
        TID(id)
    }

    fn sqrt_op(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|&x| (x + 1e-8).sqrt()).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Sqrt { a }, is_param: false,
        });
        TID(id)
    }

    fn pow_op(&mut self, a: TID, n: f32) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|&x| x.powf(n)).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Pow { a, n }, is_param: false,
        });
        TID(id)
    }

    fn clamp_op(&mut self, a: TID, lo: f32, hi: f32) -> TID {
        let data = vec_clamp(&self.nodes[a.0].data, lo, hi);
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::Clamp { a, lo, hi }, is_param: false,
        });
        TID(id)
    }

    fn one_minus(&mut self, a: TID) -> TID {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|x| 1.0 - x).collect();
        let len = data.len();
        let (r, c) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; len], rows: r, cols: c,
            op: TensorOp::OneMinus { a }, is_param: false,
        });
        TID(id)
    }

    /// [m,n] + [1,n] broadcast bias add
    fn broadcast_add(&mut self, a: TID, bias: TID) -> TID {
        let rows = self.nodes[a.0].rows;
        let cols = self.nodes[a.0].cols;
        let ad = &self.nodes[a.0].data;
        let bd = &self.nodes[bias.0].data;
        let mut out = ad.clone();
        for i in 0..rows {
            let off = i * cols;
            for j in 0..cols { out[off + j] += bd[j]; }
        }
        let len = out.len();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; len], rows, cols,
            op: TensorOp::BroadcastAdd { a, bias, rows, cols }, is_param: false,
        });
        TID(id)
    }

    /// [m,n] * [1,n] broadcast element multiply
    fn broadcast_mul(&mut self, a: TID, sc: TID) -> TID {
        let rows = self.nodes[a.0].rows;
        let cols = self.nodes[a.0].cols;
        let ad = &self.nodes[a.0].data;
        let sd = &self.nodes[sc.0].data;
        let mut out = ad.clone();
        for i in 0..rows {
            let off = i * cols;
            for j in 0..cols { out[off + j] *= sd[j]; }
        }
        let len = out.len();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; len], rows, cols,
            op: TensorOp::BroadcastMul { a, scale: sc, rows, cols }, is_param: false,
        });
        TID(id)
    }

    /// Reduce mean along cols: [m,n] -> [m,1]
    fn reduce_mean_cols(&mut self, a: TID) -> TID {
        let rows = self.nodes[a.0].rows;
        let cols = self.nodes[a.0].cols;
        let ad = &self.nodes[a.0].data;
        let inv = 1.0 / cols as f32;
        let mut out = vec![0.0; rows];
        for i in 0..rows {
            let off = i * cols;
            let mut s = 0.0;
            for j in 0..cols { s += ad[off + j]; }
            out[i] = s * inv;
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; rows], rows, cols: 1,
            op: TensorOp::ReduceMeanCols { a, rows, cols }, is_param: false,
        });
        TID(id)
    }

    /// Expand [m,1] -> [m,n]
    fn expand_col(&mut self, a: TID, cols: usize) -> TID {
        let rows = self.nodes[a.0].rows;
        let ad = &self.nodes[a.0].data;
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            let v = ad[i];
            let off = i * cols;
            for j in 0..cols { out[off + j] = v; }
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; rows * cols], rows, cols,
            op: TensorOp::ExpandCol { a, rows, cols }, is_param: false,
        });
        TID(id)
    }

    fn transpose(&mut self, a: TID) -> TID {
        let rows = self.nodes[a.0].rows;
        let cols = self.nodes[a.0].cols;
        let ad = &self.nodes[a.0].data;
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = ad[i * cols + j];
            }
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: out, grad: vec![0.0; rows * cols], rows: cols, cols: rows,
            op: TensorOp::Transpose { a, rows, cols }, is_param: false,
        });
        TID(id)
    }

    fn row_slice(&mut self, a: TID, row: usize) -> TID {
        let cols = self.nodes[a.0].cols;
        let off = row * cols;
        let data = self.nodes[a.0].data[off..off + cols].to_vec();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data, grad: vec![0.0; cols], rows: 1, cols,
            op: TensorOp::RowSlice { a, row, cols }, is_param: false,
        });
        TID(id)
    }

    fn sum_all(&mut self, a: TID) -> TID {
        let s: f32 = self.nodes[a.0].data.iter().sum();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: vec![s], grad: vec![0.0], rows: 1, cols: 1,
            op: TensorOp::SumAll { a }, is_param: false,
        });
        TID(id)
    }

    /// Fused softmax cross-entropy: [1, vocab] + target -> scalar loss
    fn softmax_ce(&mut self, logits: TID, target_idx: usize) -> TID {
        let vocab = self.nodes[logits.0].cols;
        let ld = &self.nodes[logits.0].data;
        let max_l = vec_max(ld);
        let mut sum_exp = 0.0f32;
        for j in 0..vocab { sum_exp += (ld[j] - max_l).exp(); }
        let loss = sum_exp.ln() + max_l - ld[target_idx];
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            data: vec![loss], grad: vec![0.0], rows: 1, cols: 1,
            op: TensorOp::SoftmaxCE { logits, target_idx, vocab }, is_param: false,
        });
        TID(id)
    }

    // ------------------------------------------------------------------
    // Backward — vectorized gradient propagation
    // ------------------------------------------------------------------

    fn backward(&mut self, loss: TID) {
        self.nodes[loss.0].grad = vec![1.0];
        let n = self.nodes.len();

        for i in (0..n).rev() {
            let has_grad = self.nodes[i].grad.iter().any(|&g| g != 0.0);
            if !has_grad { continue; }

            let op = self.nodes[i].op.clone();
            match op {
                TensorOp::None => {}

                TensorOp::MatMul { a, b, m, k, n: nn } => {
                    let og = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    let b_d = self.nodes[b.0].data.clone();
                    // dA = dOut @ B^T  (grad_a += og[m,nn] * b_d^T[nn,k])
                    {
                        let mut da_buf = vec![0.0f32; m * k];
                        unsafe {
                            matrixmultiply::sgemm(
                                m, nn, k,
                                1.0,
                                og.as_ptr(), nn as isize, 1,     // dOut row-major [m, nn]
                                b_d.as_ptr(), 1, nn as isize,    // B^T: read B[k,nn] with swapped strides
                                0.0,
                                da_buf.as_mut_ptr(), k as isize, 1,
                            );
                        }
                        vec_add_inplace(&mut self.nodes[a.0].grad, &da_buf);
                    }
                    // dB = A^T @ dOut  (grad_b += a_d^T[k,m] * og[m,nn])
                    {
                        let mut db_buf = vec![0.0f32; k * nn];
                        unsafe {
                            matrixmultiply::sgemm(
                                k, m, nn,
                                1.0,
                                a_d.as_ptr(), 1, k as isize,     // A^T: read A[m,k] with swapped strides
                                og.as_ptr(), nn as isize, 1,     // dOut row-major [m, nn]
                                0.0,
                                db_buf.as_mut_ptr(), nn as isize, 1,
                            );
                        }
                        vec_add_inplace(&mut self.nodes[b.0].grad, &db_buf);
                    }
                }

                TensorOp::Add { a, b } => {
                    let g = self.nodes[i].grad.clone();
                    vec_add_inplace(&mut self.nodes[a.0].grad, &g);
                    vec_add_inplace(&mut self.nodes[b.0].grad, &g);
                }

                TensorOp::Sub { a, b } => {
                    let g = self.nodes[i].grad.clone();
                    vec_add_inplace(&mut self.nodes[a.0].grad, &g);
                    for (bg, &og) in self.nodes[b.0].grad.iter_mut().zip(g.iter()) { *bg -= og; }
                }

                TensorOp::Mul { a, b } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    let b_d = self.nodes[b.0].data.clone();
                    for j in 0..g.len() {
                        self.nodes[a.0].grad[j] += g[j] * b_d[j];
                        self.nodes[b.0].grad[j] += g[j] * a_d[j];
                    }
                }

                TensorOp::DivElem { a, b } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    let b_d = self.nodes[b.0].data.clone();
                    for j in 0..g.len() {
                        let bv = b_d[j] + 1e-8;
                        self.nodes[a.0].grad[j] += g[j] / bv;
                        self.nodes[b.0].grad[j] -= g[j] * a_d[j] / (bv * bv);
                    }
                }

                TensorOp::Scale { a, s } => {
                    let g = self.nodes[i].grad.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] * s; }
                }

                TensorOp::AddScalar { a, .. } => {
                    let g = self.nodes[i].grad.clone();
                    vec_add_inplace(&mut self.nodes[a.0].grad, &g);
                }

                TensorOp::ScalarDiv { a, n: dv } => {
                    let g = self.nodes[i].grad.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] / dv; }
                }

                TensorOp::Sigmoid { a } => {
                    let g = self.nodes[i].grad.clone();
                    let od = self.nodes[i].data.clone();
                    for j in 0..g.len() {
                        let s = od[j];
                        self.nodes[a.0].grad[j] += g[j] * s * (1.0 - s);
                    }
                }

                TensorOp::Tanh { a } => {
                    let g = self.nodes[i].grad.clone();
                    let od = self.nodes[i].data.clone();
                    for j in 0..g.len() {
                        let t = od[j];
                        self.nodes[a.0].grad[j] += g[j] * (1.0 - t * t);
                    }
                }

                TensorOp::SiLU { a } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    for j in 0..g.len() {
                        let x = a_d[j];
                        let s = 1.0 / (1.0 + (-x).exp());
                        self.nodes[a.0].grad[j] += g[j] * (s * (1.0 + x * (1.0 - s)));
                    }
                }

                TensorOp::Exp { a } => {
                    let g = self.nodes[i].grad.clone();
                    let od = self.nodes[i].data.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] * od[j]; }
                }

                TensorOp::Log { a } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] / (a_d[j] + 1e-8); }
                }

                TensorOp::Square { a } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] * 2.0 * a_d[j]; }
                }

                TensorOp::Sqrt { a } => {
                    let g = self.nodes[i].grad.clone();
                    let od = self.nodes[i].data.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] * 0.5 / od[j]; }
                }

                TensorOp::Pow { a, n: pw } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] += g[j] * pw * a_d[j].powf(pw - 1.0); }
                }

                TensorOp::Neg { a } => {
                    let g = self.nodes[i].grad.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] -= g[j]; }
                }

                TensorOp::Clamp { a, lo, hi } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    for j in 0..g.len() {
                        if a_d[j] >= lo && a_d[j] <= hi { self.nodes[a.0].grad[j] += g[j]; }
                    }
                }

                TensorOp::OneMinus { a } => {
                    let g = self.nodes[i].grad.clone();
                    for j in 0..g.len() { self.nodes[a.0].grad[j] -= g[j]; }
                }

                TensorOp::BroadcastAdd { a, bias, rows: rr, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    vec_add_inplace(&mut self.nodes[a.0].grad, &g);
                    for ii in 0..rr {
                        let off = ii * cc;
                        for j in 0..cc { self.nodes[bias.0].grad[j] += g[off + j]; }
                    }
                }

                TensorOp::BroadcastMul { a, scale: sc, rows: rr, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    let a_d = self.nodes[a.0].data.clone();
                    let s_d = self.nodes[sc.0].data.clone();
                    for ii in 0..rr {
                        let off = ii * cc;
                        for j in 0..cc {
                            self.nodes[a.0].grad[off + j] += g[off + j] * s_d[j];
                            self.nodes[sc.0].grad[j] += g[off + j] * a_d[off + j];
                        }
                    }
                }

                TensorOp::ReduceMeanCols { a, rows: rr, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    let inv = 1.0 / cc as f32;
                    for ii in 0..rr {
                        let off = ii * cc;
                        let gv = g[ii] * inv;
                        for j in 0..cc { self.nodes[a.0].grad[off + j] += gv; }
                    }
                }

                TensorOp::ExpandCol { a, rows: rr, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    for ii in 0..rr {
                        let off = ii * cc;
                        let mut s = 0.0;
                        for j in 0..cc { s += g[off + j]; }
                        self.nodes[a.0].grad[ii] += s;
                    }
                }

                TensorOp::Transpose { a, rows: rr, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    for ii in 0..cc {
                        for jj in 0..rr {
                            self.nodes[a.0].grad[jj * cc + ii] += g[ii * rr + jj];
                        }
                    }
                }

                TensorOp::RowSlice { a, row, cols: cc } => {
                    let g = self.nodes[i].grad.clone();
                    let off = row * cc;
                    for j in 0..cc { self.nodes[a.0].grad[off + j] += g[j]; }
                }

                TensorOp::SumAll { a } => {
                    let gv = self.nodes[i].grad[0];
                    for v in self.nodes[a.0].grad.iter_mut() { *v += gv; }
                }

                TensorOp::SoftmaxCE { logits, target_idx, vocab } => {
                    let gv = self.nodes[i].grad[0];
                    let ld = &self.nodes[logits.0].data;
                    let max_l = vec_max(ld);
                    let exps: Vec<f32> = ld.iter().map(|&x| (x - max_l).exp()).collect();
                    let sum_e: f32 = exps.iter().sum();
                    for j in 0..vocab {
                        let prob = exps[j] / sum_e;
                        let tg = if j == target_idx { prob - 1.0 } else { prob };
                        self.nodes[logits.0].grad[j] += gv * tg;
                    }
                }
            }
        }
    }
}

// ============================================================================
// PARAMETER COLLECTION for optimizer & checkpointing
// ============================================================================

struct ParamSet {
    ids: Vec<TID>,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl ParamSet {
    fn new(ids: Vec<TID>, g: &Graph) -> Self {
        let m: Vec<Vec<f32>> = ids.iter().map(|&t| vec![0.0; g.nodes[t.0].data.len()]).collect();
        let v: Vec<Vec<f32>> = ids.iter().map(|&t| vec![0.0; g.nodes[t.0].data.len()]).collect();
        ParamSet { ids, m, v }
    }
}

// ============================================================================
// ADAMW OPTIMIZER — vectorized
// ============================================================================

struct AdamW {
    lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, t: usize,
}

impl AdamW {
    fn new(lr: f32, wd: f32) -> Self {
        AdamW { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: wd, t: 0 }
    }

    fn step(&mut self, g: &mut Graph, ps: &mut ParamSet) {
        self.t += 1;
        let t = self.t as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (idx, &tid) in ps.ids.iter().enumerate() {
            let node = &mut g.nodes[tid.0];
            let len = node.data.len();
            let pm = &mut ps.m[idx];
            let pv = &mut ps.v[idx];
            for j in 0..len {
                let grad = node.grad[j];
                if grad == 0.0 && pm[j] == 0.0 { continue; }
                pm[j] = self.beta1 * pm[j] + (1.0 - self.beta1) * grad;
                pv[j] = self.beta2 * pv[j] + (1.0 - self.beta2) * grad * grad;
                let m_hat = pm[j] / bc1;
                let v_hat = pv[j] / bc2;
                node.data[j] -= self.lr * self.weight_decay * node.data[j]
                    + self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

// ============================================================================
// EMBEDDING
// ============================================================================

struct Embedding { weight: TID, vocab_size: usize, embed_dim: usize }

impl Embedding {
    fn new(vocab_size: usize, embed_dim: usize, g: &mut Graph, rng: &mut impl Rng) -> Self {
        let scale = (1.0 / embed_dim as f32).sqrt();
        let weight = g.param(rand_normal(vocab_size * embed_dim, scale, rng), vocab_size, embed_dim);
        Embedding { weight, vocab_size, embed_dim }
    }

    fn forward(&self, idx: usize, g: &mut Graph) -> TID {
        g.row_slice(self.weight, idx.min(self.vocab_size - 1))
    }

    fn param_ids(&self) -> Vec<TID> { vec![self.weight] }
}

// ============================================================================
// RMSNORM
// ============================================================================

struct RMSNorm { weight: TID, dim: usize, eps: f32 }

impl RMSNorm {
    fn new(dim: usize, g: &mut Graph) -> Self {
        RMSNorm { weight: g.param(vec![1.0; dim], 1, dim), dim, eps: 1e-6 }
    }

    fn forward(&self, x: TID, g: &mut Graph) -> TID {
        let xsq = g.square(x);
        let msq = g.reduce_mean_cols(xsq);
        let msq_eps = g.add_scalar(msq, self.eps);
        let rms = g.sqrt_op(msq_eps);
        let cols = g.cols(x);
        let rms_exp = g.expand_col(rms, cols);
        let inv = g.pow_op(rms_exp, -1.0);
        let norm = g.mul(x, inv);
        g.broadcast_mul(norm, self.weight)
    }

    fn param_ids(&self) -> Vec<TID> { vec![self.weight] }
}

// ============================================================================
// SWIGLU EXPERT
// ============================================================================

struct SwiGLUExpert { w_gate: TID, w_up: TID, w_down: TID }

impl SwiGLUExpert {
    fn new(dim: usize, hdim: usize, g: &mut Graph, rng: &mut impl Rng) -> Self {
        let s = (2.0 / (dim + hdim) as f32).sqrt();
        SwiGLUExpert {
            w_gate: g.param(rand_normal(dim * hdim, s, rng), dim, hdim),
            w_up:   g.param(rand_normal(dim * hdim, s, rng), dim, hdim),
            w_down: g.param(rand_normal(hdim * dim, s, rng), hdim, dim),
        }
    }

    fn forward(&self, x: TID, g: &mut Graph) -> TID {
        let mg = g.matmul(x, self.w_gate);
        let gate = g.silu(mg);
        let up = g.matmul(x, self.w_up);
        let gated = g.mul(gate, up);
        g.matmul(gated, self.w_down)
    }

    fn param_ids(&self) -> Vec<TID> { vec![self.w_gate, self.w_up, self.w_down] }
}

// ============================================================================
// MIXTURE OF EXPERTS — parallel expert forward via rayon
// ============================================================================

struct MoE {
    num_experts: usize, top_k: usize,
    experts: Vec<SwiGLUExpert>,
    router: Option<TID>,
    norm: RMSNorm,
}

impl MoE {
    fn new(dim: usize, num_experts: usize, top_k: usize, hdim: usize,
           g: &mut Graph, rng: &mut impl Rng) -> Self {
        let top_k = top_k.min(num_experts.saturating_sub(1));
        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts { experts.push(SwiGLUExpert::new(dim, hdim, g, rng)); }
        let router = if num_experts > 1 {
            let rs = (2.0 / (dim + num_experts - 1) as f32).sqrt();
            Some(g.param(rand_normal(dim * (num_experts - 1), rs, rng), dim, num_experts - 1))
        } else { None };
        MoE { num_experts, top_k, experts, router, norm: RMSNorm::new(dim, g) }
    }

    fn forward(&self, x: TID, g: &mut Graph) -> TID {
        let normed = self.norm.forward(x, g);

        // Always-on expert 0
        let mut output = self.experts[0].forward(normed, g);

        if self.num_experts > 1 && self.top_k > 0 {
            let router = self.router.unwrap();
            let rl = g.matmul(normed, router);
            let rl_data = g.data(rl).to_vec();
            let _ne = self.num_experts - 1;

            // Softmax routing probabilities
            let mx = vec_max(&rl_data);
            let exps: Vec<f32> = rl_data.iter().map(|x| (x - mx).exp()).collect();
            let se: f32 = exps.iter().sum::<f32>() + 1e-8;
            let probs: Vec<f32> = exps.iter().map(|e| e / se).collect();

            // Top-k selection
            let mut idx_w: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            idx_w.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let selected: Vec<(usize, f32)> = idx_w.into_iter().take(self.top_k).collect();

            if !selected.is_empty() {
                let ws: f32 = selected.iter().map(|(_, w)| w).sum();
                let total = ws + 1.0;
                let ao_w = 1.0 / (total + 1e-8);
                output = g.scale(output, ao_w);

                // ---- PARALLEL expert forward passes via rayon ----
                // We read normed data, fan out to experts in parallel,
                // collect results, then insert them back into the graph.
                let normed_data = g.data(normed).to_vec();
                let normed_rows = g.nodes[normed.0].rows;
                let normed_cols = g.nodes[normed.0].cols;

                // Collect expert weight data needed for forward
                let expert_specs: Vec<(usize, f32, Vec<f32>, usize, usize,
                                       Vec<f32>, usize, usize,
                                       Vec<f32>, usize, usize)> = selected.iter().map(|&(eidx, w)| {
                    let ei = eidx + 1;
                    let e = &self.experts[ei];
                    let nw = w / (total + 1e-8);
                    let wg_d = g.data(e.w_gate).to_vec();
                    let wg_r = g.nodes[e.w_gate.0].rows;
                    let wg_c = g.nodes[e.w_gate.0].cols;
                    let wu_d = g.data(e.w_up).to_vec();
                    let wu_r = g.nodes[e.w_up.0].rows;
                    let wu_c = g.nodes[e.w_up.0].cols;
                    let wd_d = g.data(e.w_down).to_vec();
                    let wd_r = g.nodes[e.w_down.0].rows;
                    let wd_c = g.nodes[e.w_down.0].cols;
                    (eidx, nw, wg_d, wg_r, wg_c, wu_d, wu_r, wu_c, wd_d, wd_r, wd_c)
                }).collect();

                // Run experts in parallel
                let expert_outputs: Vec<(usize, f32, Vec<f32>)> = expert_specs.into_par_iter().map(
                    |(eidx, nw, wg_d, _wg_r, wg_c, wu_d, _wu_r, wu_c, wd_d, _wd_r, wd_c)| {
                        let hdim = wg_c;
                        let dim = wd_c;
                        let m = normed_rows;
                        // gate = silu(normed @ w_gate)
                        let mut gate = vec![0.0; m * hdim];
                        for i in 0..m {
                            for j in 0..hdim {
                                let mut s = 0.0;
                                for kk in 0..normed_cols { s += normed_data[i * normed_cols + kk] * wg_d[kk * hdim + j]; }
                                let sig = 1.0 / (1.0 + (-s).exp());
                                gate[i * hdim + j] = s * sig;
                            }
                        }
                        // up = normed @ w_up
                        let mut up = vec![0.0; m * wu_c];
                        for i in 0..m {
                            for j in 0..wu_c {
                                let mut s = 0.0;
                                for kk in 0..normed_cols { s += normed_data[i * normed_cols + kk] * wu_d[kk * wu_c + j]; }
                                up[i * wu_c + j] = s;
                            }
                        }
                        // gated = gate * up
                        let gated: Vec<f32> = gate.iter().zip(up.iter()).map(|(a, b)| a * b).collect();
                        // out = gated @ w_down
                        let mut out = vec![0.0; m * dim];
                        for i in 0..m {
                            for j in 0..dim {
                                let mut s = 0.0;
                                for kk in 0..hdim { s += gated[i * hdim + kk] * wd_d[kk * dim + j]; }
                                out[i * dim + j] = s;
                            }
                        }
                        (eidx, nw, out)
                    }
                ).collect();

                // Insert parallel results back into graph (needed for backprop through experts)
                for (eidx, nw, _out_data) in expert_outputs {
                    let ei = eidx + 1;
                    // We need graph nodes for backprop, so replay the forward through the graph
                    let expert_out = self.experts[ei].forward(normed, g);
                    let weighted = g.scale(expert_out, nw);
                    output = g.add(output, weighted);
                }
            }
        }
        output
    }

    fn param_ids(&self) -> Vec<TID> {
        let mut ids = self.norm.param_ids();
        for e in &self.experts { ids.extend(e.param_ids()); }
        if let Some(r) = self.router { ids.push(r); }
        ids
    }
}

// ============================================================================
// MODERN GRU CELL WITH MULTI-HEAD RWKV
// ============================================================================

#[derive(Clone)]
struct KVState {
    num_state: Vec<f32>,
    den_state: Vec<f32>,
    x_prev: Vec<f32>,
}

struct GRUCellResult { h: TID, kv_state: Option<KVState> }

struct ModernGRUCell {
    hidden_size: usize, use_rwkv: bool, num_heads: usize, head_dim: usize,
    wz: TID, uz: TID, bz: TID,
    wr: TID, ur: TID, br: TID,
    wh: TID, uh: TID, bh: TID,
    wk: Option<TID>, wv: Option<TID>, wo: Option<TID>,
    time_decay_proj: Option<TID>, time_first_proj: Option<TID>,
    time_decay: Option<TID>, time_first: Option<TID>,
    mix_k: Option<TID>, mix_v: Option<TID>, mix_r: Option<TID>,
    input_norm: RMSNorm, hidden_norm: RMSNorm,
}

impl ModernGRUCell {
    fn new(input_size: usize, hidden_size: usize, use_rwkv: bool, num_heads: usize,
           g: &mut Graph, rng: &mut impl Rng) -> Self {
        let head_dim = hidden_size / num_heads;
        let si = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let sh = (2.0 / (hidden_size * 2) as f32).sqrt();

        let wz = g.param(rand_normal(input_size * hidden_size, si, rng), input_size, hidden_size);
        let uz = g.param(rand_normal(hidden_size * hidden_size, sh, rng), hidden_size, hidden_size);
        let bz = g.param(vec![0.0; hidden_size], 1, hidden_size);
        let wr = g.param(rand_normal(input_size * hidden_size, si, rng), input_size, hidden_size);
        let ur = g.param(rand_normal(hidden_size * hidden_size, sh, rng), hidden_size, hidden_size);
        let br = g.param(vec![0.0; hidden_size], 1, hidden_size);
        let wh = g.param(rand_normal(input_size * hidden_size, si, rng), input_size, hidden_size);
        let uh = g.param(rand_normal(hidden_size * hidden_size, sh, rng), hidden_size, hidden_size);
        let bh = g.param(vec![0.0; hidden_size], 1, hidden_size);

        let (wk, wv, wo, tdp, tfp, td, tf, mk, mv, mr) = if use_rwkv {
            let wk = g.param(rand_normal(hidden_size * hidden_size, sh, rng), hidden_size, hidden_size);
            let wv = g.param(rand_normal(hidden_size * hidden_size, sh, rng), hidden_size, hidden_size);
            let wo = g.param(rand_normal(hidden_size * hidden_size, sh, rng), hidden_size, hidden_size);
            let tdp = g.param(rand_normal(hidden_size * hidden_size, sh * 0.1, rng), hidden_size, hidden_size);
            let tfp = g.param(rand_normal(hidden_size * hidden_size, sh * 0.1, rng), hidden_size, hidden_size);
            let mut td_d = vec![0.0; num_heads * head_dim];
            for h in 0..num_heads { for d in 0..head_dim {
                td_d[h * head_dim + d] = -0.3 - (h as f32 * 0.1) - rng.gen::<f32>() * 0.4;
            }}
            let td = g.param(td_d, 1, hidden_size);
            let mut tf_d = vec![0.0; num_heads * head_dim];
            for h in 0..num_heads { for d in 0..head_dim {
                tf_d[h * head_dim + d] = 0.3 + rng.gen::<f32>() * 0.2;
            }}
            let tf = g.param(tf_d, 1, hidden_size);
            let mk_d: Vec<f32> = (0..hidden_size).map(|_| 0.5 + (rng.gen::<f32>() - 0.5) * 0.1).collect();
            let mv_d: Vec<f32> = (0..hidden_size).map(|_| 0.5 + (rng.gen::<f32>() - 0.5) * 0.1).collect();
            let mr_d: Vec<f32> = (0..hidden_size).map(|_| 0.5 + (rng.gen::<f32>() - 0.5) * 0.1).collect();
            (Some(wk), Some(wv), Some(wo), Some(tdp), Some(tfp),
             Some(td), Some(tf),
             Some(g.param(mk_d, 1, hidden_size)),
             Some(g.param(mv_d, 1, hidden_size)),
             Some(g.param(mr_d, 1, hidden_size)))
        } else {
            (None, None, None, None, None, None, None, None, None, None)
        };

        ModernGRUCell {
            hidden_size, use_rwkv, num_heads, head_dim,
            wz, uz, bz, wr, ur, br, wh, uh, bh,
            wk, wv, wo, time_decay_proj: tdp, time_first_proj: tfp,
            time_decay: td, time_first: tf, mix_k: mk, mix_v: mv, mix_r: mr,
            input_norm: RMSNorm::new(input_size, g),
            hidden_norm: RMSNorm::new(hidden_size, g),
        }
    }

    fn init_kv_state(&self) -> Option<KVState> {
        if !self.use_rwkv { return None; }
        Some(KVState {
            num_state: vec![0.0; self.hidden_size],
            den_state: vec![0.0; self.hidden_size],
            x_prev: vec![0.0; self.hidden_size],
        })
    }

    fn forward(&self, x: TID, h_prev: TID, kv_state: Option<KVState>, g: &mut Graph) -> GRUCellResult {
        let xn = self.input_norm.forward(x, g);
        let hn = self.hidden_norm.forward(h_prev, g);

        // z = sigmoid(xn @ Wz + hn @ Uz + bz)
        let z_wx = g.matmul(xn, self.wz);
        let z_uh = g.matmul(hn, self.uz);
        let z_pre = g.add(z_wx, z_uh);
        let z_pre = g.broadcast_add(z_pre, self.bz);
        let z = g.sigmoid(z_pre);

        // r = sigmoid(xn @ Wr + hn @ Ur + br)
        let r_wx = g.matmul(xn, self.wr);
        let r_uh = g.matmul(hn, self.ur);
        let r_pre = g.add(r_wx, r_uh);
        let r_pre = g.broadcast_add(r_pre, self.br);
        let r = g.sigmoid(r_pre);

        // h_cand = tanh(xn @ Wh + (r*hn) @ Uh + bh)
        let rh = g.mul(r, hn);
        let h_wx = g.matmul(xn, self.wh);
        let h_uh = g.matmul(rh, self.uh);
        let h_pre = g.add(h_wx, h_uh);
        let h_pre = g.broadcast_add(h_pre, self.bh);
        let mut h_cand = g.tanh_op(h_pre);

        // RWKV enhancement — fully in-graph for gradient flow
        let new_kv = if self.use_rwkv {
            let kv = kv_state.unwrap_or_else(|| KVState {
                num_state: vec![0.0; self.hidden_size],
                den_state: vec![0.0; self.hidden_size],
                x_prev: vec![0.0; self.hidden_size],
            });

            let hs = self.hidden_size;
            let x_prev_tid = g.constant(kv.x_prev.clone(), 1, hs);

            // Token-shift mixing
            let mk = self.mix_k.unwrap();
            let mv = self.mix_v.unwrap();
            let mr = self.mix_r.unwrap();
            let omk = g.one_minus(mk);
            let omv = g.one_minus(mv);
            let omr = g.one_minus(mr);

            let sk_1 = g.mul(omk, x_prev_tid);
            let sk_2 = g.mul(mk, h_cand);
            let sk = g.add(sk_1, sk_2);

            let sv_1 = g.mul(omv, x_prev_tid);
            let sv_2 = g.mul(mv, h_cand);
            let sv = g.add(sv_1, sv_2);

            let sr_1 = g.mul(omr, x_prev_tid);
            let sr_2 = g.mul(mr, h_cand);
            let sr = g.add(sr_1, sr_2);

            // K, V projections — remain in graph for gradient flow
            let k = g.matmul(sk, self.wk.unwrap());
            let v = g.matmul(sv, self.wv.unwrap());

            // Dynamic decay and bonus projections — remain in graph
            let da_mm = g.matmul(h_cand, self.time_decay_proj.unwrap());
            let da = g.tanh_op(da_mm);
            let fa_mm = g.matmul(h_cand, self.time_first_proj.unwrap());
            let fa = g.tanh_op(fa_mm);

            // Clamp k for numerical stability
            let k_clamped = g.clamp_op(k, -10.0, 10.0);

            // w_dyn = time_decay + da * 0.5  [1, hs]
            let da_scaled = g.scale(da, 0.5);
            let w_dyn = g.add(self.time_decay.unwrap(), da_scaled);

            // u_dyn = time_first + fa * 0.3  [1, hs]
            let fa_scaled = g.scale(fa, 0.3);
            let u_dyn = g.add(self.time_first.unwrap(), fa_scaled);

            // exp_k = exp(k_clamped), exp_uk = exp(u_dyn + k_clamped)
            let exp_k = g.exp_op(k_clamped);
            let uk = g.add(u_dyn, k_clamped);
            let exp_uk = g.exp_op(uk);

            // decay = exp(-exp(w_dyn)) — proper double-exp decay in (0,1)
            let exp_w = g.exp_op(w_dyn);
            let neg_exp_w = g.neg(exp_w);
            let decay = g.exp_op(neg_exp_w);

            // KV state from previous time step (constants — no BPTT through KV state)
            let pn_tid = g.constant(kv.num_state.clone(), 1, hs);
            let pd_tid = g.constant(kv.den_state.clone(), 1, hs);

            // WKV attention: wkv = (exp_uk * v + pn) / (exp_uk + pd)
            let num_term = g.mul(exp_uk, v);
            let numerator = g.add(num_term, pn_tid);
            let denominator = g.add(exp_uk, pd_tid);
            let wkv_tid = g.div_elem(numerator, denominator);

            // State update for next time step (outside graph — recurrent state)
            let decay_data = g.data(decay).to_vec();
            let exp_k_data = g.data(exp_k).to_vec();
            let v_raw = g.data(v).to_vec();
            let mut new_num = vec![0.0f32; hs];
            let mut new_den = vec![0.0f32; hs];
            for i in 0..hs {
                new_num[i] = decay_data[i] * kv.num_state[i] + exp_k_data[i] * v_raw[i];
                new_den[i] = decay_data[i] * kv.den_state[i] + exp_k_data[i];
            }

            let new_state = KVState {
                num_state: new_num, den_state: new_den,
                x_prev: g.data(h_cand).to_vec(),
            };

            // Gate with receptance and project output
            let r_sig = g.sigmoid(sr);
            let gated = g.mul(r_sig, wkv_tid);
            let rwkv_out = g.matmul(gated, self.wo.unwrap());
            h_cand = g.add(h_cand, rwkv_out);

            Some(new_state)
        } else { kv_state };

        // GRU update: h = (1-z)*h_prev + z*h_cand
        let omz = g.one_minus(z);
        let h_part1 = g.mul(omz, h_prev);
        let h_part2 = g.mul(z, h_cand);
        let h_new = g.add(h_part1, h_part2);

        GRUCellResult { h: h_new, kv_state: new_kv }
    }

    fn param_ids(&self) -> Vec<TID> {
        let mut ids = vec![self.wz, self.uz, self.bz, self.wr, self.ur, self.br, self.wh, self.uh, self.bh];
        ids.extend(self.input_norm.param_ids());
        ids.extend(self.hidden_norm.param_ids());
        if self.use_rwkv {
            ids.extend([self.wk.unwrap(), self.wv.unwrap(), self.wo.unwrap(),
                self.time_decay_proj.unwrap(), self.time_first_proj.unwrap(),
                self.time_decay.unwrap(), self.time_first.unwrap(),
                self.mix_k.unwrap(), self.mix_v.unwrap(), self.mix_r.unwrap()]);
        }
        ids
    }
}

// ============================================================================
// MODERN RNN LAYER
// ============================================================================

struct ModernRNNLayer {
    cell: ModernGRUCell, moe: Option<MoE>,
    has_residual: bool, proj_in: Option<TID>,
}

impl ModernRNNLayer {
    fn new(input_size: usize, hidden_size: usize, use_mlp: bool, use_rwkv: bool,
           num_experts: usize, top_k: usize, num_heads: usize,
           g: &mut Graph, rng: &mut impl Rng) -> Self {
        let cell = ModernGRUCell::new(input_size, hidden_size, use_rwkv, num_heads, g, rng);
        let moe = if use_mlp {
            Some(MoE::new(hidden_size, num_experts, top_k, hidden_size * 2, g, rng))
        } else { None };
        let has_residual = input_size == hidden_size;
        let proj_in = if !has_residual {
            let s = (2.0 / (input_size + hidden_size) as f32).sqrt();
            Some(g.param(rand_normal(input_size * hidden_size, s, rng), input_size, hidden_size))
        } else { None };
        ModernRNNLayer { cell, moe, has_residual, proj_in }
    }

    fn init_kv_state(&self) -> Option<KVState> { self.cell.init_kv_state() }

    fn forward(&self, x: TID, h_prev: TID, kv: Option<KVState>, g: &mut Graph) -> (TID, Option<KVState>) {
        let xr = if let Some(p) = self.proj_in { g.matmul(x, p) } else { x };
        let res = self.cell.forward(x, h_prev, kv, g);
        let mut out = if self.has_residual || self.proj_in.is_some() { g.add(res.h, xr) } else { res.h };
        if let Some(ref moe) = self.moe {
            let mo = moe.forward(out, g);
            out = g.add(out, mo);
        }
        (out, res.kv_state)
    }

    fn param_ids(&self) -> Vec<TID> {
        let mut ids = self.cell.param_ids();
        if let Some(ref moe) = self.moe { ids.extend(moe.param_ids()); }
        if let Some(p) = self.proj_in { ids.push(p); }
        ids
    }

    fn has_moe(&self) -> bool { self.moe.is_some() }
}

// ============================================================================
// MODERN RNN MODEL
// ============================================================================

struct HiddenState { hs: Vec<Vec<f32>>, kv_states: Vec<Option<KVState>> }
struct ForwardResult { outputs: Vec<TID>, final_hidden: HiddenState }

struct ModernRNN {
    vocab_size: usize, embed_size: usize, hidden_size: usize,
    num_layers: usize, fast_mode: bool, use_rwkv: bool,
    num_experts: usize, top_k: usize, num_heads: usize,
    embedding: Embedding, layers: Vec<ModernRNNLayer>,
    output_norm: RMSNorm, output_proj: TID,
}

impl ModernRNN {
    fn new(vocab_size: usize, embed_size: usize, hidden_size: usize, num_layers: usize,
           fast_mode: bool, use_rwkv: bool, num_experts: usize, top_k: usize, num_heads: usize,
           g: &mut Graph, rng: &mut impl Rng) -> Self {
        let embedding = Embedding::new(vocab_size, embed_size, g, rng);
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let inp = if i == 0 { embed_size } else { hidden_size };
            layers.push(ModernRNNLayer::new(inp, hidden_size, !fast_mode, use_rwkv, num_experts, top_k, num_heads, g, rng));
        }
        let output_norm = RMSNorm::new(hidden_size, g);
        let os = (2.0 / (hidden_size + vocab_size) as f32).sqrt();
        let output_proj = g.param(rand_normal(hidden_size * vocab_size, os, rng), hidden_size, vocab_size);
        ModernRNN { vocab_size, embed_size, hidden_size, num_layers, fast_mode, use_rwkv,
            num_experts, top_k, num_heads, embedding, layers, output_norm, output_proj }
    }

    fn param_ids(&self) -> Vec<TID> {
        let mut ids = self.embedding.param_ids();
        for l in &self.layers { ids.extend(l.param_ids()); }
        ids.extend(self.output_norm.param_ids());
        ids.push(self.output_proj);
        ids
    }

    fn total_params(&self, g: &Graph) -> usize {
        self.param_ids().iter().map(|&t| g.nodes[t.0].data.len()).sum()
    }

    fn active_params(&self, g: &Graph) -> usize {
        let mut c = 0usize;
        for &t in &self.embedding.param_ids() { c += g.nodes[t.0].data.len(); }
        for layer in &self.layers {
            if layer.has_moe() {
                let cp: usize = layer.cell.param_ids().iter().map(|&t| g.nodes[t.0].data.len()).sum();
                let d = self.hidden_size; let hd = d * 2;
                let ep = 3 * hd * d;
                c += cp + d + d * (self.num_experts - 1) + (1 + self.top_k) * ep;
                if let Some(p) = layer.proj_in { c += g.nodes[p.0].data.len(); }
            } else {
                for &t in &layer.param_ids() { c += g.nodes[t.0].data.len(); }
            }
        }
        for &t in &self.output_norm.param_ids() { c += g.nodes[t.0].data.len(); }
        c += g.nodes[self.output_proj.0].data.len();
        c
    }

    fn forward(&self, xs: &[usize], h_init: Option<HiddenState>, g: &mut Graph) -> ForwardResult {
        let mut outputs = Vec::with_capacity(xs.len());
        let hs = self.hidden_size;
        // Initialize hidden state TIDs — only the INITIAL state is a detached constant
        let mut lh_tids: Vec<TID> = match &h_init {
            Some(h) => h.hs.iter().map(|hv| g.constant(hv.clone(), 1, hs)).collect(),
            None => (0..self.num_layers).map(|_| g.constant(vec![0.0; hs], 1, hs)).collect(),
        };
        let mut kvs: Vec<Option<KVState>> = match h_init {
            Some(h) => h.kv_states,
            None => self.layers.iter().map(|l| l.init_kv_state()).collect(),
        };

        for &idx in xs {
            let mut h = self.embedding.forward(idx, g);
            let mut new_h_tids = Vec::with_capacity(self.num_layers);
            let mut new_kvs = Vec::with_capacity(self.num_layers);

            for i in 0..self.num_layers {
                let hp = lh_tids[i];  // TID stays in graph — gradients flow through time
                let li = h;
                let kv = kvs[i].take();
                let (out, nkv) = self.layers[i].forward(li, hp, kv, g);
                h = out;
                if i > 0 && self.num_layers > 1 {
                    let sc = g.scale(li, 0.5);
                    h = g.add(h, sc);
                }
                new_h_tids.push(h);  // Keep as TID — enables BPTT
                new_kvs.push(nkv);
            }
            lh_tids = new_h_tids;
            kvs = new_kvs;

            let norm = self.output_norm.forward(h, g);
            outputs.push(g.matmul(norm, self.output_proj));
        }

        // Extract final hidden states as data for cross-call persistence (generation)
        let final_hs: Vec<Vec<f32>> = lh_tids.iter().map(|&t| g.data(t).to_vec()).collect();
        ForwardResult { outputs, final_hidden: HiddenState { hs: final_hs, kv_states: kvs } }
    }

    fn cross_entropy_loss(&self, outputs: &[TID], targets: &[usize], g: &mut Graph) -> TID {
        let n = outputs.len();
        let mut total = g.softmax_ce(outputs[0], targets[0]);
        for i in 1..n { let ce = g.softmax_ce(outputs[i], targets[i]); total = g.add(total, ce); }
        g.scalar_div(total, n as f32)
    }

    fn clip_grad(g: &mut Graph, pids: &[TID], max_norm: f32) -> f32 {
        let mut tn = 0.0f32;
        for &t in pids { for &gv in &g.nodes[t.0].grad { tn += gv * gv; } }
        tn = tn.sqrt();
        if tn > max_norm {
            let s = max_norm / tn;
            for &t in pids { vec_scale_inplace(&mut g.nodes[t.0].grad, s); }
        }
        tn
    }

    fn save_checkpoint(&self, fp: &str, meta: serde_json::Value, step: usize, g: &Graph, ps: &ParamSet) {
        let total_len: usize = ps.ids.iter().map(|&t| g.nodes[t.0].data.len()).sum();
        let mut data_vec = Vec::with_capacity(total_len);
        let mut m_vec = Vec::with_capacity(total_len);
        let mut v_vec = Vec::with_capacity(total_len);
        for (idx, &tid) in ps.ids.iter().enumerate() {
            data_vec.extend_from_slice(&g.nodes[tid.0].data);
            m_vec.extend_from_slice(&ps.m[idx]);
            v_vec.extend_from_slice(&ps.v[idx]);
        }
        let cp = Checkpoint {
            version: "3.0-bin".to_string(), timestamp: iso_timestamp(),
            metadata: Some(serde_json::to_string(&meta).unwrap_or_default()),
            optimizer_step: step,
            config: CheckpointConfig {
                vocab_size: self.vocab_size, embed_size: self.embed_size,
                hidden_size: self.hidden_size, num_layers: self.num_layers,
                fast_mode: self.fast_mode, use_rwkv: self.use_rwkv,
                num_experts: self.num_experts, top_k: self.top_k, num_heads: self.num_heads,
            },
            param_data: data_vec, param_m: m_vec, param_v: v_vec,
        };
        let bytes = bincode::serialize(&cp).unwrap();
        fs::write(fp, &bytes).unwrap();
        println!("Saved binary checkpoint to {} ({:.2} MB)", fp, bytes.len() as f64 / 1_048_576.0);
    }

    fn load_checkpoint(&self, fp: &str, g: &mut Graph, ps: &mut ParamSet) -> Checkpoint {
        let bytes = fs::read(fp).unwrap();
        let cp: Checkpoint = bincode::deserialize(&bytes).unwrap();
        let mut off = 0;
        for (idx, &tid) in ps.ids.iter().enumerate() {
            let len = g.nodes[tid.0].data.len();
            g.nodes[tid.0].data[..len].copy_from_slice(&cp.param_data[off..off + len]);
            ps.m[idx][..len].copy_from_slice(&cp.param_m[off..off + len]);
            ps.v[idx][..len].copy_from_slice(&cp.param_v[off..off + len]);
            off += len;
        }
        println!("Loaded binary checkpoint from {} ({} params, {:.2} MB)", fp, off, bytes.len() as f64 / 1_048_576.0);
        cp
    }

    fn peek_checkpoint(fp: &str) -> CheckpointInfo {
        let bytes = fs::read(fp).unwrap();
        let cp: Checkpoint = bincode::deserialize(&bytes).unwrap();
        CheckpointInfo { config: cp.config, metadata: cp.metadata.clone(), timestamp: cp.timestamp,
            param_count: cp.param_data.len(), optimizer_step: cp.optimizer_step }
    }
}

// ============================================================================
// CHECKPOINT SERIALIZATION
// ============================================================================

#[derive(Serialize, Deserialize, Clone)]
struct Checkpoint {
    version: String,
    timestamp: String,
    metadata: Option<String>,
    optimizer_step: usize,
    config: CheckpointConfig,
    param_data: Vec<f32>,
    param_m: Vec<f32>,
    param_v: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone)]
struct CheckpointConfig {
    vocab_size: usize,
    embed_size: usize,
    hidden_size: usize,
    num_layers: usize,
    fast_mode: bool,
    use_rwkv: bool,
    num_experts: usize,
    top_k: usize,
    num_heads: usize,
}

struct CheckpointInfo {
    config: CheckpointConfig, metadata: Option<String>,
    timestamp: String, param_count: usize, optimizer_step: usize,
}

fn iso_timestamp() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let (secs, ms) = (now.as_secs(), now.subsec_millis());
    let s = secs as i64;
    let (days, tod) = (s / 86400, s % 86400);
    let (h, m, sec) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let mut y = 1970i64; let mut rd = days;
    loop {
        let diy = if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 { 366 } else { 365 };
        if rd < diy { break; } rd -= diy; y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let md = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0;
    for d in md { if rd < d { break; } rd -= d; mo += 1; }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z", y, mo + 1, rd + 1, h, m, sec, ms)
}

fn rand_normal(n: usize, scale: f32, rng: &mut impl Rng) -> Vec<f32> {
    (0..n).map(|_| {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * scale
    }).collect()
}

// ============================================================================
// DATA PREPARATION
// ============================================================================

struct TrainingData {
    inputs: Vec<usize>, targets: Vec<usize>, vocab_size: usize,
    char_to_idx: HashMap<String, usize>, idx_to_char: HashMap<usize, String>,
    text: String,
}

fn prepare_char_data(text: &str) -> TrainingData {
    let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
    chars.sort();
    let c2i: HashMap<String, usize> = chars.iter().enumerate().map(|(i, c)| (c.to_string(), i)).collect();
    let i2c: HashMap<usize, String> = chars.iter().enumerate().map(|(i, c)| (i, c.to_string())).collect();
    let vs = chars.len();
    let tc: Vec<char> = text.chars().collect();
    let mut inp = Vec::with_capacity(tc.len() - 1);
    let mut tgt = Vec::with_capacity(tc.len() - 1);
    for i in 0..tc.len() - 1 {
        inp.push(*c2i.get(&tc[i].to_string()).unwrap_or(&0));
        tgt.push(*c2i.get(&tc[i + 1].to_string()).unwrap_or(&0));
    }
    TrainingData { inputs: inp, targets: tgt, vocab_size: vs, char_to_idx: c2i, idx_to_char: i2c, text: text.to_string() }
}

// ============================================================================
// BPE TOKENIZER
// ============================================================================

struct BPETokenizer {
    target_vocab_size: usize, merges: Vec<(usize, usize)>,
    vocab: HashMap<usize, String>, inverse_vocab: HashMap<String, usize>,
    special_tokens: HashMap<String, usize>, trained: bool,
}

impl BPETokenizer {
    fn new(vs: usize) -> Self {
        let st: HashMap<String, usize> = [("<PAD>".into(), 0), ("<UNK>".into(), 1), ("<BOS>".into(), 2), ("<EOS>".into(), 3)].into_iter().collect();
        let mut t = BPETokenizer { target_vocab_size: vs, merges: Vec::new(), vocab: HashMap::new(), inverse_vocab: HashMap::new(), special_tokens: st, trained: false };
        t.init_base(); t
    }
    fn init_base(&mut self) {
        self.vocab.clear(); self.inverse_vocab.clear();
        for (tok, &id) in &self.special_tokens { self.vocab.insert(id, tok.clone()); self.inverse_vocab.insert(tok.clone(), id); }
        let bo = self.special_tokens.len();
        for i in 0..256u16 { let s = ((i as u8) as char).to_string(); self.vocab.insert(bo + i as usize, s.clone()); self.inverse_vocab.insert(s, bo + i as usize); }
    }
    fn vocab_size(&self) -> usize { self.vocab.len() }
    fn text_to_bytes(&self, text: &str) -> Vec<usize> {
        let mut t = Vec::new();
        for ch in text.chars() {
            if (ch as u32) < 256 { if let Some(&id) = self.inverse_vocab.get(&ch.to_string()) { t.push(id); } }
            else { for b in ch.to_string().into_bytes() { if let Some(&id) = self.inverse_vocab.get(&(b as char).to_string()) { t.push(id); } } }
        }
        t
    }
    fn count_pairs(tl: &[Vec<usize>]) -> HashMap<(usize, usize), usize> {
        let mut c = HashMap::new();
        for ts in tl { for i in 0..ts.len().saturating_sub(1) { *c.entry((ts[i], ts[i+1])).or_insert(0) += 1; } }
        c
    }
    fn merge_pair(tl: &[Vec<usize>], a: usize, b: usize, nid: usize) -> Vec<Vec<usize>> {
        tl.iter().map(|ts| { let mut r = Vec::new(); let mut i = 0; while i < ts.len() { if i < ts.len()-1 && ts[i]==a && ts[i+1]==b { r.push(nid); i+=2; } else { r.push(ts[i]); i+=1; } } r }).collect()
    }
    fn train(&mut self, text: &str, min_freq: usize, verbose: bool) {
        let tm = self.target_vocab_size.saturating_sub(self.vocab.len());
        if verbose { println!("\n{}\n   BPE TOKENIZER TRAINING\n{}\nText: {} chars | Target: {} | Base: {} | Max merges: {} | Min freq: {}\n", "=".repeat(70), "=".repeat(70), text.len(), self.target_vocab_size, self.vocab.len(), tm, min_freq); }
        let words: Vec<&str> = text.split_inclusive(char::is_whitespace).collect();
        let mut tl: Vec<Vec<usize>> = words.iter().map(|w| self.text_to_bytes(w)).collect();
        let start = Instant::now(); let mut md = 0;
        while md < tm {
            let pc = Self::count_pairs(&tl);
            if pc.is_empty() { break; }
            let (&bp, &bc) = pc.iter().max_by_key(|(_,&c)| c).unwrap();
            if bc < min_freq { break; }
            let (a, b) = bp; let nid = self.vocab.len();
            let ns = format!("{}{}", self.vocab.get(&a).unwrap_or(&String::new()), self.vocab.get(&b).unwrap_or(&String::new()));
            self.vocab.insert(nid, ns.clone()); self.inverse_vocab.insert(ns.clone(), nid); self.merges.push((a, b));
            tl = Self::merge_pair(&tl, a, b, nid); md += 1;
            if verbose && (md % 100 == 0 || md <= 10) { println!("Merge {:4}: \"{}\" (count: {}) | Vocab: {} | {:.1}s", md, ns.replace('\n', "\\n"), bc, self.vocab.len(), start.elapsed().as_secs_f64()); }
        }
        self.trained = true;
        if verbose { println!("\nDone in {:.1}s | Vocab: {} | Merges: {}\n", start.elapsed().as_secs_f64(), self.vocab.len(), self.merges.len()); }
    }
    fn encode(&self, text: &str) -> Vec<usize> {
        let mut ts = self.text_to_bytes(text);
        if !self.trained && self.merges.is_empty() { return ts; }
        for &(a, b) in &self.merges {
            let ns = format!("{}{}", self.vocab.get(&a).unwrap_or(&String::new()), self.vocab.get(&b).unwrap_or(&String::new()));
            let nid = match self.inverse_vocab.get(&ns) { Some(&id) => id, None => continue };
            let mut r = Vec::new(); let mut i = 0;
            while i < ts.len() { if i < ts.len()-1 && ts[i]==a && ts[i+1]==b { r.push(nid); i+=2; } else { r.push(ts[i]); i+=1; } }
            ts = r;
        }
        ts
    }
    fn decode(&self, ts: &[usize]) -> String { ts.iter().filter_map(|&id| self.vocab.get(&id)).filter(|s| !s.starts_with('<')).cloned().collect() }
    fn save(&self, fp: &str) {
        let d = BPEFile { version: "1.0".into(), type_: "BPE".into(), timestamp: iso_timestamp(), target_vocab_size: self.target_vocab_size, vocab_size: self.vocab.len(), merge_count: self.merges.len(), special_tokens: self.special_tokens.clone(), merges: self.merges.iter().map(|&(a,b)| vec![a,b]).collect(), vocab: self.vocab.iter().map(|(&id, s)| (id, s.clone())).collect() };
        fs::write(fp, serde_json::to_string_pretty(&d).unwrap()).unwrap();
        println!("Saved BPE tokenizer to {} (vocab={}, merges={})", fp, self.vocab.len(), self.merges.len());
    }
    fn load_from_file(fp: &str) -> Self {
        let d: BPEFile = serde_json::from_str(&fs::read_to_string(fp).unwrap()).unwrap();
        let mut t = BPETokenizer::new(d.target_vocab_size);
        t.merges = d.merges.iter().map(|v| (v[0], v[1])).collect(); t.special_tokens = d.special_tokens; t.trained = true;
        t.vocab.clear(); t.inverse_vocab.clear();
        for (id, s) in d.vocab { t.inverse_vocab.insert(s.clone(), id); t.vocab.insert(id, s); }
        println!("Loaded BPE from {} (vocab={}, merges={})", fp, t.vocab.len(), t.merges.len()); t
    }
    fn prepare_data(&self, text: &str) -> TrainingData {
        let ts = self.encode(text);
        let mut c2i = HashMap::new(); let mut i2c = HashMap::new();
        for (&id, s) in &self.vocab { c2i.insert(s.clone(), id); i2c.insert(id, s.clone()); }
        TrainingData { inputs: ts[..ts.len()-1].to_vec(), targets: ts[1..].to_vec(), vocab_size: self.vocab.len(), char_to_idx: c2i, idx_to_char: i2c, text: text.to_string() }
    }
    fn print_stats(&self) {
        println!("\n=== BPE Stats === Vocab: {} | Merges: {}", self.vocab.len(), self.merges.len());
        let mut bl: HashMap<usize, Vec<(usize, String)>> = HashMap::new();
        for (&id, s) in &self.vocab { if !s.starts_with('<') { bl.entry(s.len()).or_default().push((id, s.clone())); } }
        let mut ls: Vec<usize> = bl.keys().copied().collect(); ls.sort_by(|a, b| b.cmp(a));
        for l in ls.iter().take(5) { let ts = &bl[l]; let sa: Vec<String> = ts.iter().take(5).map(|(_, s)| format!("\"{}\"", s.replace('\n', "\\n"))).collect();
            println!("  Len {}: {}{}", l, sa.join(", "), if ts.len() > 5 { format!(" ...({} total)", ts.len()) } else { String::new() });
        }
    }
}

#[derive(Serialize, Deserialize)]
struct BPEFile {
    version: String, #[serde(rename = "type")] type_: String, timestamp: String,
    #[serde(rename = "targetVocabSize")] target_vocab_size: usize,
    #[serde(rename = "vocabSize")] vocab_size: usize,
    #[serde(rename = "mergeCount")] merge_count: usize,
    #[serde(rename = "specialTokens")] special_tokens: HashMap<String, usize>,
    merges: Vec<Vec<usize>>, vocab: Vec<(usize, String)>,
}

fn train_bpe(config: &Config) {
    println!("\n{}\n   BPE TOKENIZER TRAINING MODE\n{}\n", "=".repeat(70), "=".repeat(70));
    
    // Priority: bpe_training_file > bpe_training_text > training_file > training_text
    let mut tt = config.training_text.clone();
    
    if !config.training_file.is_empty() && std::path::Path::new(&config.training_file).exists() {
        if let Ok(content) = fs::read_to_string(&config.training_file) { tt = content; }
    }
    
    if !config.bpe_training_text.is_empty() { tt = config.bpe_training_text.clone(); }
    
    if !config.bpe_training_file.is_empty() && std::path::Path::new(&config.bpe_training_file).exists() { 
        if let Ok(content) = fs::read_to_string(&config.bpe_training_file) { tt = content; }
    }

    println!("Text: {} chars | Target: {} | Output: {}\n", tt.len(), config.bpe_vocab_size, config.bpe_save_path);
    let mut tok = BPETokenizer::new(config.bpe_vocab_size);
    tok.train(&tt, config.bpe_min_frequency, true); tok.save(&config.bpe_save_path); tok.print_stats();
    let test = &tt[..tt.len().min(100)]; let enc = tok.encode(test); let dec = tok.decode(&enc);
    println!("\n=== Test === Original: {} chars | Encoded: {} tokens | Compression: {:.2}x | Match: {}", test.len(), enc.len(), test.len() as f32 / enc.len() as f32, if dec == test { "YES ✓" } else { "NO ✗" });
}

// ============================================================================
// TRAINING
// ============================================================================

fn train(config: &Config) -> (Graph, ModernRNN, TrainingData, ParamSet) {
    println!("\n{}\n   MODERN RNN — VECTORIZED AUTODIFF + SIMD + RAYON (2026)\n{}\n", "=".repeat(70), "=".repeat(70));
    let mut rng = rand::thread_rng();

    let mut training_text = config.training_text.clone();
    if !config.training_file.is_empty() && std::path::Path::new(&config.training_file).exists() {
        match fs::read_to_string(&config.training_file) {
            Ok(content) => {
                training_text = content;
                println!("Loaded training text from file: {} ({} chars)", config.training_file, training_text.len());
            }
            Err(e) => println!("Error reading training file: {}", e),
        }
    }

    let (data, _tok) = if config.use_bpe {
        let tok = if !config.bpe_load_path.is_empty() && std::path::Path::new(&config.bpe_load_path).exists() {
            BPETokenizer::load_from_file(&config.bpe_load_path)
        } else { let mut t = BPETokenizer::new(config.bpe_vocab_size); t.train(&training_text, config.bpe_min_frequency, true); t.save(&config.bpe_save_path); t };
        let d = tok.prepare_data(&training_text); println!("Tokenizer: BPE (vocab={})", tok.vocab_size()); (d, Some(tok))
    } else { (prepare_char_data(&training_text), None) };

    println!("\nConfig: hidden={} layers={} embed={} heads={} experts={} topk={} lr={} wd={} epochs={} batch={} seq={} fast={} rwkv={} vocab={}",
        config.hidden_size, config.num_layers, config.embed_size, config.num_heads, config.num_experts, config.top_k,
        config.learning_rate, config.weight_decay, config.epochs, config.batch_size, config.seq_length, config.fast_mode, config.use_rwkv, data.vocab_size);
    println!("Engine: Vectorized Tensor Autodiff + SIMD + Rayon\n");

    let mut g = Graph::new();
    let mut start_epoch = 0usize;
    let rnn = if !config.load_checkpoint.is_empty() && std::path::Path::new(&config.load_checkpoint).exists() {
        let info = ModernRNN::peek_checkpoint(&config.load_checkpoint);
        let c = &info.config;
        let r = ModernRNN::new(data.vocab_size, c.embed_size, c.hidden_size, c.num_layers, c.fast_mode, c.use_rwkv, c.num_experts, c.top_k, c.num_heads, &mut g, &mut rng);
        if let Some(ref ms) = info.metadata {
            if let Ok(m) = serde_json::from_str::<serde_json::Value>(ms) {
                if let Some(e) = m.get("epoch").and_then(|e| e.as_u64()) { start_epoch = e as usize; }
            }
        }
        r
    } else {
        ModernRNN::new(data.vocab_size, config.embed_size, config.hidden_size, config.num_layers, config.fast_mode, config.use_rwkv, config.num_experts, config.top_k, config.num_heads, &mut g, &mut rng)
    };

    g.freeze_params();
    let mut ps = ParamSet::new(rnn.param_ids(), &g);
    let mut opt = AdamW::new(config.learning_rate, config.weight_decay);

    if !config.load_checkpoint.is_empty() && std::path::Path::new(&config.load_checkpoint).exists() {
        let cp = rnn.load_checkpoint(&config.load_checkpoint, &mut g, &mut ps);
        if cp.optimizer_step > 0 { opt.t = cp.optimizer_step; println!("Restored optimizer step: {}", opt.t); }
    }

    let tp = rnn.total_params(&g); let ap = rnn.active_params(&g);
    println!("Total params: {} | Active: {} ({:.1}%)", tp, ap, ap as f32 / tp as f32 * 100.0);

    let mut seqs: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let mut i = 0;
    while i < data.inputs.len().saturating_sub(config.seq_length) {
        let e = (i + config.seq_length).min(data.inputs.len());
        seqs.push((data.inputs[i..e].to_vec(), data.targets[i..e].to_vec()));
        i += config.seq_length;
    }

    println!("\nTraining...\n");
    let t0 = Instant::now();
    let mut best_loss = f32::INFINITY;
    let pids = rnn.param_ids();

    for epoch in start_epoch..config.epochs {
        let mut el = 0.0; let mut nb = 0; let mut tgn = 0.0;
        let mut shuf = seqs.clone();
        for i in (1..shuf.len()).rev() { let j = rng.gen_range(0..=i); shuf.swap(i, j); }

        let mut b = 0;
        while b < shuf.len() {
            let be = (b + config.batch_size).min(shuf.len());
            let batch = &shuf[b..be]; let bs = batch.len();
            g.zero_grad();
            let mut bl = 0.0;
            for (xs, ts) in batch {
                g.reset();
                let res = rnn.forward(xs, None, &mut g);
                let loss = rnn.cross_entropy_loss(&res.outputs, ts, &mut g);
                let scaled = g.scalar_div(loss, bs as f32);
                g.backward(scaled);
                bl += g.data(loss)[0];
            }
            tgn += ModernRNN::clip_grad(&mut g, &pids, 1.0);
            opt.step(&mut g, &mut ps);
            el += bl / bs as f32; nb += 1; b = be;
        }

        let al = el / nb.max(1) as f32;
        if (epoch + 1) % config.log_every == 0 || epoch == start_epoch {
            println!("Epoch {:4} | Loss: {:.6} | \u{2207}: {:.4} | Time: {:.1}s", epoch + 1, al, tgn / nb.max(1) as f32, t0.elapsed().as_secs_f64());
        }
        if config.save_every > 0 && (epoch + 1) % config.save_every == 0 {
            let cp = config.save_path.replace(".bin", &format!("_epoch{}.bin", epoch + 1));
            rnn.save_checkpoint(&cp, serde_json::json!({"epoch": epoch+1, "loss": al}), opt.t, &g, &ps);
            println!("  [Checkpoint: {}]", cp);
        }
        if al < best_loss { best_loss = al; }
    }

    let tt = t0.elapsed().as_secs_f64();
    println!("\nDone in {:.1}s | Best loss: {:.6}\n", tt, best_loss);
    if config.save_on_complete {
        rnn.save_checkpoint(&config.save_path, serde_json::json!({"epoch": config.epochs, "loss": best_loss, "time": format!("{:.1}", tt), "completed": true}), opt.t, &g, &ps);
    }
    println!("Generating sample...\n");
    g.reset();
    generate(&rnn, &data, config, &mut g, &mut rng);
    (g, rnn, data, ps)
}

// ============================================================================
// GENERATION
// ============================================================================

fn generate(rnn: &ModernRNN, data: &TrainingData, config: &Config, g: &mut Graph, rng: &mut impl Rng) -> String {
    let len = config.gen_length;
    let fc = &data.text[..1];
    let mut ci = *data.char_to_idx.get(fc).unwrap_or(&0);
    let mut gen = fc.to_string();
    let mut hs: Option<HiddenState> = None;
    for _ in 0..len {
        g.reset();
        let r = rnn.forward(&[ci], hs, g);
        hs = Some(r.final_hidden);
        let logits = g.data(r.outputs[0]).to_vec();
        let sc: Vec<f32> = logits.iter().map(|l| l / config.temperature).collect();
        let mx = vec_max(&sc);
        let ex: Vec<f32> = sc.iter().map(|l| (l - mx).exp()).collect();
        let sm: f32 = ex.iter().sum();
        let pr: Vec<f32> = ex.iter().map(|e| e / sm).collect();
        let mut rv: f32 = rng.gen(); let mut idx = 0;
        for j in 0..pr.len() { rv -= pr[j]; if rv <= 0.0 { idx = j; break; } }
        ci = idx;
        gen.push_str(data.idx_to_char.get(&idx).map(|s| s.as_str()).unwrap_or(" "));
    }
    println!("Temperature: {}\nGenerated ({} tokens):\n{}\n{}\n{}\n", config.temperature, len, "-".repeat(70), gen, "-".repeat(70));
    gen
}

// ============================================================================
// RLVR — Reinforcement Learning with Verifiable Rewards
// ============================================================================

struct RLVR { lr: f32, num_samples: usize, max_gen_len: usize, temperature: f32 }
struct RLSample { tokens: Vec<String>, log_probs: Vec<f32>, text: String }
struct GRPOResult { mean_reward: f32, max_reward: f32, #[allow(dead_code)] loss: f32, samples: Vec<(String, f32)> }

impl RLVR {
    fn new(config: &Config) -> Self { RLVR { lr: config.rl_learning_rate, num_samples: 4, max_gen_len: 20, temperature: 1.0 } }

    fn gen_with_lp(&self, model: &ModernRNN, data: &TrainingData, prompt: &str, max_len: usize, g: &mut Graph) -> RLSample {
        let mut toks = Vec::new(); let mut lps = Vec::new();
        let mut hs: Option<HiddenState> = None; let mut ct = prompt.to_string();
        let mut rng = rand::thread_rng();
        for _ in 0..max_len {
            let ci = *data.char_to_idx.get(&ct).unwrap_or(&0);
            g.reset();
            let r = model.forward(&[ci], hs, g); hs = Some(r.final_hidden);
            let logits = g.data(r.outputs[0]).to_vec();
            let mx = vec_max(&logits);
            let sc: Vec<f32> = logits.iter().map(|l| (l - mx) / self.temperature).collect();
            let ex: Vec<f32> = sc.iter().map(|x| x.exp()).collect();
            let sm: f32 = ex.iter().sum::<f32>() + 1e-8;
            let pr: Vec<f32> = ex.iter().map(|e| e / sm).collect();
            let mut rv: f32 = rng.gen(); let mut si = 0;
            for j in 0..pr.len() { rv -= pr[j]; if rv <= 0.0 { si = j; break; } }
            lps.push((pr[si] + 1e-8).ln());
            let sc = data.idx_to_char.get(&si).cloned().unwrap_or(" ".into());
            toks.push(sc.clone()); ct = sc;
            if ct == "\n" || ct == "|" { break; }
        }
        RLSample { tokens: toks.clone(), log_probs: lps, text: toks.join("") }
    }

    fn compute_reward(prompt: &str, gen: &str) -> f32 {
        if prompt.contains('=') {
            let pt = prompt.trim_end_matches('=');
            let mut op = '+'; let mut pos = 0;
            for (i, c) in pt.char_indices() { if c == '+' || c == '-' || c == '*' { op = c; pos = i; break; } }
            let a: i64 = pt[..pos].parse().unwrap_or(0); let b: i64 = pt[pos+1..].parse().unwrap_or(0);
            let exp = match op { '+' => a+b, '-' => a-b, '*' => a*b, _ => return 0.0 };
            if gen.trim().parse::<i64>().unwrap_or(i64::MIN) == exp { 1.0 } else {
                let ec: Vec<char> = exp.to_string().chars().collect(); let gc: Vec<char> = gen.trim().chars().collect();
                let mut m = 0; for i in 0..ec.len().min(gc.len()) { if ec[i] == gc[i] { m += 1; } }
                m as f32 / ec.len() as f32 * 0.5
            }
        } else if prompt.contains('>') {
            let inp = prompt.trim_end_matches('>').trim(); let exp: String = inp.chars().rev().collect();
            if gen.trim() == exp { 1.0 } else {
                let gc: Vec<char> = gen.chars().collect(); let ec: Vec<char> = exp.chars().collect();
                let mut m = 0; for i in 0..gc.len().min(ec.len()) { if gc[i] == ec[i] { m += 1; } }
                m as f32 / ec.len() as f32 * 0.5
            }
        } else if prompt.contains(':') {
            let exp = prompt.trim_end_matches(':').trim();
            if gen.trim() == exp { 1.0 } else {
                let gc: Vec<char> = gen.chars().collect(); let ec: Vec<char> = exp.chars().collect();
                let mut m = 0; for i in 0..gc.len().min(ec.len()) { if gc[i] == ec[i] { m += 1; } }
                m as f32 / ec.len() as f32
            }
        } else {
            let gc: Vec<char> = gen.chars().collect(); let ec: Vec<char> = prompt.chars().collect();
            let mut m = 0; for i in 0..gc.len().min(ec.len()) { if gc[i] == ec[i] { m += 1; } else { break; } }
            m as f32 / ec.len() as f32
        }
    }

    fn grpo(&self, model: &ModernRNN, data: &TrainingData, prompt: &str, g: &mut Graph, opt: &mut AdamW, ps: &mut ParamSet) -> GRPOResult {
        // 1. Collect samples and rewards
        let mut samps = Vec::new(); let mut rews = Vec::new();
        for _ in 0..self.num_samples {
            let s = self.gen_with_lp(model, data, prompt, self.max_gen_len, g);
            let r = Self::compute_reward(prompt, &s.text); samps.push(s); rews.push(r);
        }

        // 2. Compute per-sample advantages (z-scored)
        let mr: f32 = rews.iter().sum::<f32>() / rews.len() as f32;
        let var: f32 = rews.iter().map(|r| (r - mr).powi(2)).sum::<f32>() / rews.len() as f32;
        let std = (var + 1e-8).sqrt();
        let advs: Vec<f32> = rews.iter().map(|r| (r - mr) / std).collect();

        // 3. Accumulate per-sample policy gradients
        let pids = model.param_ids();
        g.zero_grad();
        let mut total_loss = 0.0f32;

        for (si, samp) in samps.iter().enumerate() {
            if samp.tokens.is_empty() || advs[si].abs() < 1e-8 { continue; }

            // Build input sequence: [prompt_token, gen_0, gen_1, ...]
            let prompt_idx = *data.char_to_idx.get(prompt).unwrap_or(&0);
            let gen_indices: Vec<usize> = samp.tokens.iter()
                .map(|t| *data.char_to_idx.get(t).unwrap_or(&0))
                .collect();

            let mut input_seq = vec![prompt_idx];
            if gen_indices.len() > 1 {
                input_seq.extend(&gen_indices[..gen_indices.len() - 1]);
            }

            g.reset();
            let res = model.forward(&input_seq, None, g);
            let n_tgt = gen_indices.len().min(res.outputs.len());
            if n_tgt == 0 { continue; }

            let loss = model.cross_entropy_loss(&res.outputs[..n_tgt], &gen_indices[..n_tgt], g);
            // Scale: -advantage / num_samples (negative because minimize loss = maximize reward)
            let scaled = g.scale(loss, -advs[si] / self.num_samples as f32);
            g.backward(scaled);
            total_loss += g.data(loss)[0] * advs[si];
        }

        // 4. Single optimizer step on accumulated gradients
        ModernRNN::clip_grad(g, &pids, 1.0);
        opt.step(g, ps);

        let mxr = rews.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        GRPOResult { mean_reward: mr, max_reward: mxr, loss: total_loss, samples: samps.into_iter().enumerate().map(|(i, s)| (s.text, rews[i])).collect() }
    }

    fn gen_prompts(task: &str, n: usize, rng: &mut impl Rng) -> Vec<String> {
        let cv: Vec<char> = "abcdefghij".chars().collect();
        (0..n).map(|_| match task {
            "arithmetic" => { let (a, b) = (rng.gen_range(0..10), rng.gen_range(0..10)); let ops = ['+', '-', '*']; format!("{}{}{}=", a, ops[rng.gen_range(0..3)], b) }
            "reverse" => { let l = 2 + rng.gen_range(0..4); format!("{}>", (0..l).map(|_| cv[rng.gen_range(0..cv.len())]).collect::<String>()) }
            "copy" => { let l = 2 + rng.gen_range(0..5); format!("{}:", (0..l).map(|_| cv[rng.gen_range(0..cv.len())]).collect::<String>()) }
            _ => "hello".into(),
        }).collect()
    }

    fn train_rl(&self, model: &ModernRNN, data: &TrainingData, task: &str, eps: usize, le: usize, g: &mut Graph, opt: &mut AdamW, ps: &mut ParamSet) {
        println!("\n{}\n   RLVR: {} TASK\n{}\nEpisodes: {} | Samples: {} | LR: {} | Method: GRPO\n", "=".repeat(70), task.to_uppercase(), "=".repeat(70), eps, self.num_samples, self.lr);
        let t0 = Instant::now(); let mut tr = 0.0; let mut sc = 0; let mut rng = rand::thread_rng();
        for ep in 0..eps {
            let p = Self::gen_prompts(task, 1, &mut rng); g.reset();
            let r = self.grpo(model, data, &p[0], g, opt, ps);
            tr += r.mean_reward; if r.max_reward >= 1.0 { sc += 1; }
            if (ep + 1) % le == 0 || ep == 0 {
                let best = r.samples.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
                println!("Ep {:4} | R: {:.3} | Avg: {:.3} | Success: {:.1}% | {:.1}s\n  \"{}\" -> \"{}\" (r={:.2})",
                    ep+1, r.mean_reward, tr/(ep+1) as f32, sc as f32/(ep+1) as f32*100.0, t0.elapsed().as_secs_f64(), p[0], best.0, best.1);
            }
        }
        println!("\nRL done in {:.1}s | Success: {:.1}%\n", t0.elapsed().as_secs_f64(), sc as f32 / eps as f32 * 100.0);
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let config = Config::from_args();
    println!("\nUsage: rnn-rust [options]");
    println!("Options: --hiddenSize=32 --epochs=500 --learningRate=0.001 etc.");
    println!("RL: --doRL=true --rlTask=copy --rlEpisodes=100");
    println!("Checkpoint: --savePath=./model.bin --saveEvery=50 --loadCheckpoint=./model.bin");
    println!("BPE: --trainBPE=true --bpeVocabSize=512 --useBPE=true --bpeLoadPath=./tokenizer.json\n");

    if config.train_bpe { train_bpe(&config); return; }

    let (mut g, rnn, data, mut ps) = train(&config);

    if config.do_rl {
        let rlvr = RLVR::new(&config);
        let mut rl_opt = AdamW::new(config.rl_learning_rate, 0.001);
        rlvr.train_rl(&rnn, &data, &config.rl_task, config.rl_episodes, 10, &mut g, &mut rl_opt, &mut ps);
        println!("Post-RL Generation:"); g.reset();
        let mut rng = rand::thread_rng();
        generate(&rnn, &data, &config, &mut g, &mut rng);
    }
}
