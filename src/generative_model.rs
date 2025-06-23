// src/generative_model.rs

//! This module contains the `tch`-based neural network for text generation.
//! This version fully implements the Transformer-based architecture.

use crate::Frame;
use tch::{
    nn,
    nn::{ModuleT, Path},
    Kind, Tensor,
};

// --- Custom Transformer Components ---

#[derive(Debug)]
pub struct MultiHeadAttention {
    w_q: nn::Linear,
    w_k: nn::Linear,
    w_v: nn::Linear,
    w_out: nn::Linear,
    nhead: i64,
    d_k: i64,
    dropout: f64,
}
impl MultiHeadAttention {
    pub fn new(p: &Path, d_model: i64, nhead: i64, dropout: f64) -> Self {
        let d_k = d_model / nhead;
        let w_q = nn::linear(p / "w_q", d_model, d_model, Default::default());
        let w_k = nn::linear(p / "w_k", d_model, d_model, Default::default());
        let w_v = nn::linear(p / "w_v", d_model, d_model, Default::default());
        let w_out = nn::linear(p / "w_out", d_model, d_model, Default::default());
        Self { w_q, w_k, w_v, w_out, nhead, d_k, dropout }
    }
    pub fn forward_t(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>, train: bool) -> Tensor {
        let (batch_size, seq_len, _) = q.size3().unwrap();
        let q = q.apply(&self.w_q).view([batch_size, seq_len, self.nhead, self.d_k]).transpose(1, 2);
        let k = k.apply(&self.w_k).view([batch_size, seq_len, self.nhead, self.d_k]).transpose(1, 2);
        let v = v.apply(&self.w_v).view([batch_size, seq_len, self.nhead, self.d_k]).transpose(1, 2);
        let scores = q.matmul(&k.transpose(-2, -1)) / (self.d_k as f64).sqrt();
        // The mask is now correctly applied here.
        let scores = match mask { Some(mask) => scores.masked_fill(mask, f64::NEG_INFINITY), None => scores, };
        let attn_weights = scores.softmax(-1, Kind::Float).dropout(self.dropout, train);
        let context = attn_weights.matmul(&v).transpose(1, 2).contiguous().view([batch_size, seq_len, -1]);
        context.apply(&self.w_out)
    }
}

#[derive(Debug)]
pub struct FeedForward {
    linear1: nn::Linear,
    linear2: nn::Linear,
}
impl FeedForward {
    pub fn new(p: &Path, d_model: i64, dim_feedforward: i64) -> Self {
        let linear1 = nn::linear(p / "linear1", d_model, dim_feedforward, Default::default());
        let linear2 = nn::linear(p / "linear2", dim_feedforward, d_model, Default::default());
        Self { linear1, linear2 }
    }
}
impl ModuleT for FeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.linear1).relu().apply(&self.linear2).dropout(0.1, train)
    }
}

#[derive(Debug)]
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
}
impl TransformerEncoderLayer {
    pub fn new(p: &Path, d_model: i64, nhead: i64, dim_feedforward: i64, dropout_p: f64) -> Self {
        let self_attn = MultiHeadAttention::new(&(p / "self_attn"), d_model, nhead, dropout_p);
        let feed_forward = FeedForward::new(&(p / "feed_forward"), d_model, dim_feedforward);
        let norm1 = nn::layer_norm(p / "norm1", vec![d_model], Default::default());
        let norm2 = nn::layer_norm(p / "norm2", vec![d_model], Default::default());
        Self { self_attn, feed_forward, norm1, norm2 }
    }
    pub fn forward_t(&self, src: &Tensor, mask: Option<&Tensor>, train: bool) -> Tensor {
        let attn_output = self.self_attn.forward_t(src, src, src, mask, train);
        let src = (src + attn_output).apply(&self.norm1);
        let ff_output = self.feed_forward.forward_t(&src, train);
        (src + ff_output).apply(&self.norm2)
    }
}

#[derive(Debug)]
pub struct PositionalEncoding {
    pe: Tensor,
    dropout_p: f64,
}
impl PositionalEncoding {
    pub fn new(_p: &Path, d_model: i64, max_len: i64, device: tch::Device) -> Self {
        let dropout_p = 0.1;
        let pe = Tensor::zeros(&[max_len, d_model], (Kind::Float, device));
        let pos = Tensor::arange(max_len, (Kind::Float, device)).unsqueeze(1);
        let div_term = (Tensor::arange_start_step(0, d_model, 2, (Kind::Float, device))
            * -(10000.0f64.ln() / d_model as f64))
            .exp();
        pe.slice(1, 0, d_model, 2).copy_(&(&pos * &div_term).sin());
        pe.slice(1, 1, d_model, 2).copy_(&(&pos * &div_term).cos());
        Self { pe: pe.unsqueeze(0), dropout_p }
    }
}
impl ModuleT for PositionalEncoding {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + self.pe.narrow(1, 0, xs.size()[1]).to(xs.device());
        xs.dropout(self.dropout_p, train)
    }
}

#[derive(Debug)]
pub struct GenerativeModel {
    embedding: nn::Embedding,
    pos_encoder: PositionalEncoding,
    layers: Vec<TransformerEncoderLayer>,
    linear: nn::Linear,
    d_model: i64,
}

impl GenerativeModel {
    pub fn new(vs: &nn::VarStore, vocab_size: i64, d_model: i64, nhead: i64, num_layers: i64, dim_feedforward: i64) -> Self {
        let p = &vs.root();
        let device = vs.device();
        let embedding = nn::embedding(p / "embedding", vocab_size, d_model, Default::default());
        let pos_encoder = PositionalEncoding::new(&(p / "pos_encoder"), d_model, 5000, device);

        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer = TransformerEncoderLayer::new(&(p / i), d_model, nhead, dim_feedforward, 0.1);
            layers.push(layer);
        }

        let linear = nn::linear(p / "linear", d_model, vocab_size, Default::default());
        Self { embedding, pos_encoder, layers, linear, d_model }
    }

    // CORRECTED: This function now returns a boolean mask.
    fn generate_square_subsequent_mask(&self, size: i64, device: tch::Device) -> Tensor {
        Tensor::ones(&[size, size], (Kind::Bool, device)).triu(1)
    }
}

impl ModuleT for GenerativeModel {
    fn forward_t(&self, src: &Tensor, train: bool) -> Tensor {
        let seq_len = src.size()[1];
        let mask = self.generate_square_subsequent_mask(seq_len, src.device());

        let mut output = src.apply(&self.embedding) * (self.d_model as f64).sqrt();
        output = output.apply_t(&self.pos_encoder, train);

        for layer in &self.layers {
            output = layer.forward_t(&output, Some(&mask), train);
        }

        output.apply(&self.linear)
    }
}

/// The complete Adamo Language Model
pub struct AdamoLlm {
    pub frame: Frame,
    pub model: GenerativeModel,
}
impl AdamoLlm {
    pub fn new(frame: Frame, model: GenerativeModel) -> Self {
        Self { frame, model }
    }
    pub fn get_sampling_temperature(&self) -> f64 {
        const BASE_TEMP: f64 = 1.0;
        const MIN_TEMP: f64 = 0.1;
        let quality = self.frame.model().quality as f64;
        let complexity = self.frame.model().complexity as f64;
        if quality == 0.0 || complexity == 0.0 { return BASE_TEMP; }
        let coherence_factor = (1.0 + (quality / complexity)).ln();
        (BASE_TEMP / (coherence_factor + 1e-6)).max(MIN_TEMP)
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn::VarStore, Device};

    #[test]
    fn can_create_and_forward_transformer_model() {
        let vs = VarStore::new(Device::Cpu);
        let vocab_size = 1000;
        let d_model = 128;
        let nhead = 4;
        let num_layers = 2;
        let dim_feedforward = 512;

        let model = GenerativeModel::new(&vs, vocab_size, d_model, nhead, num_layers, dim_feedforward);

        let input = Tensor::f_from_slice(&[10i64, 20, 30, 40, 50]).unwrap().view((1, 5));

        let logits = model.forward_t(&input, false);

        assert_eq!(logits.size(), &[1, 5, vocab_size]);
    }
}
