// src/bin/train.rs

use adamo::{
    generative_model::GenerativeModel,
    text_processing::TextProcessor,
};
use anyhow::{anyhow, Result};
use arrow::array::{Array, ListArray, StringArray, StructArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use regex::Regex;
use std::{env, fs::{self, File}, path::Path, time::Instant};
use tch::{nn, nn::{OptimizerConfig, ModuleT}, Device, Tensor};
use std::f64::consts::PI;

// --- NEW: Define a constant for our model's context window ---
const MAX_SEQ_LEN: usize = 4096;

/// Cleans raw text from the dataset.
fn clean_text(text: &str) -> String {
    let header_regex = Regex::new(r"=\s*[^=]+\s*=").unwrap();
    let whitespace_regex = Regex::new(r"\s+").unwrap();
    let cleaned = header_regex.replace_all(text, "");
    let cleaned = whitespace_regex.replace_all(&cleaned, " ");
    cleaned.trim().to_string()
}

/// Loads articles from any supported file type.
fn load_articles_from_file(file_path: &Path) -> Result<Vec<String>> {
    let extension = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");
    match extension {
        "raw" | "txt" => { /* unchanged */
            Ok(fs::read_to_string(file_path)?.split("\n \n").map(|s| s.to_string()).collect())
        }
        "parquet" => {
            let file = File::open(file_path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let messages_column_name = "messages";
            if builder.schema().field_with_name(messages_column_name).is_err() { anyhow::bail!("Parquet file '{}' must contain a column named '{}'", file_path.display(), messages_column_name); }

            let reader = builder.build()?;
            let mut articles = Vec::new();
            for record_batch in reader {
                let record_batch = record_batch?;
                if let Some(list_array) = record_batch.column_by_name(messages_column_name).and_then(|c| c.as_any().downcast_ref::<ListArray>()) {
                    for i in 0..list_array.len() {
                        if list_array.is_valid(i) {
                            if let Some(struct_array) = list_array.value(i).as_any().downcast_ref::<StructArray>() {
                                if let Some(content_array) = struct_array.column_by_name("content").and_then(|c| c.as_any().downcast_ref::<StringArray>()) {
                                    let conversation = content_array.iter().filter_map(|val| val.map(ToString::to_string)).collect::<Vec<_>>().join(" ");
                                    articles.push(conversation);
                                }
                            }
                        }
                    }
                }
            }
            Ok(articles)
        }
        _ => { anyhow::bail!("Unsupported file type: '{}'.", extension); }
    }
}

// CORRECTED: This function now truncates long sequences.
fn text_to_batch(
    processor: &TextProcessor,
    text: &str,
    device: Device,
) -> Result<Option<(Tensor, Tensor)>> {
    let mut token_ids: Vec<i64> = processor.encode(text)?
        .get_ids()
        .iter()
        .map(|&id| id as i64)
        .collect();

    // Truncate the sequence if it's too long.
    if token_ids.len() > MAX_SEQ_LEN {
        token_ids.truncate(MAX_SEQ_LEN);
    }

    if token_ids.len() < 2 { return Ok(None); }

    let input_ids = &token_ids[..token_ids.len() - 1];
    let target_ids = &token_ids[1..];
    let xs = Tensor::f_from_slice(input_ids)?.to(device).view((1, -1));
    let ys = Tensor::f_from_slice(target_ids)?.to(device).view([-1]);
    Ok(Some((xs, ys)))
}

fn main() -> Result<()> {
    println!("--- Adamo CPU-Optimized Training (with Checkpointing) ---");
    let device = Device::Cpu;

    // --- 1. Setup ---
    let args: Vec<String> = env::args().collect();
    // CORRECTED: The output model path is no longer needed as an argument.
    if args.len() < 3 {
        eprintln!("Usage: cargo run --bin train <tokenizer.json> <file1> [file2]...");
        return Err(anyhow!("Missing required arguments"));
    }
    let tokenizer_path = &args[1];
    let training_data_paths: Vec<&String> = args.iter().skip(2).collect();

    let checkpoint_dir = "checkpoints_final";
    fs::create_dir_all(checkpoint_dir)?;

    let processor = TextProcessor::new(tokenizer_path)?;
    let vocab_size = processor.get_vocab_size() as i64;

    println!("> Loading training data...");
    let mut all_articles = Vec::new();
    for path_str in training_data_paths {
        all_articles.extend(load_articles_from_file(Path::new(path_str))?);
    }
    println!("> Loaded a total of {} articles/rows.", all_articles.len());

    // --- 2. Build the Model ---
    let vs = nn::VarStore::new(device);
    let d_model = 256;
    let nhead = 4;
    let num_layers = 4;
    let dim_feedforward = 1024;
    let generative_model = GenerativeModel::new(&vs, vocab_size, d_model, nhead, num_layers, dim_feedforward);

    let initial_lr = 1e-4;
    let mut optimizer = nn::Adam::default().build(&vs, initial_lr)?;

    println!("> Adamo Transformer Initialized for CPU optimization.");

    // --- 3. Training Loop ---
    println!("\n> Starting deep training loop...");
    let num_epochs = 30; // Let's aim for 30
    let accumulation_steps = 8;
    let clip_grad_norm = 1.0;

    let total_training_steps = num_epochs * (all_articles.len() / accumulation_steps);
    let mut current_step = 0;

    for epoch in 1..=num_epochs {
        let epoch_start_time = Instant::now();
        let mut total_loss = 0.0;
        let mut article_idx = 0;

        optimizer.zero_grad();

        let mut final_lr = 0.0;
        for article in &all_articles {
            let cleaned_article = clean_text(article);
            if let Some((xs, ys)) = text_to_batch(&processor, &cleaned_article, device)? {
                let logits = generative_model.forward_t(&xs, true);
                let loss = logits.view([-1, vocab_size]).cross_entropy_for_logits(&ys);

                total_loss += loss.double_value(&[]);
                (loss / accumulation_steps as f64).backward();
                article_idx += 1;

                if article_idx % accumulation_steps == 0 {
                    optimizer.clip_grad_norm(clip_grad_norm);
                    optimizer.step();
                    optimizer.zero_grad();
                    current_step += 1;
                    let new_lr = initial_lr * (1.0 + (PI * current_step as f64 / total_training_steps as f64).cos()) / 2.0;
                    optimizer.set_lr(new_lr);
                    final_lr = new_lr;
                }
            }
        }

        if article_idx > 0 {
            let avg_loss = total_loss / article_idx as f64;
            println!(
                "  - Epoch: {:<3} | Avg. Loss: {:.4} | LR: {:.6} | Duration: {:?}",
                epoch,
                avg_loss,
                final_lr,
                epoch_start_time.elapsed()
            );

            // CORRECTED: Re-introduced the per-epoch checkpoint saving.
            let checkpoint_path = Path::new(checkpoint_dir).join(format!("adamo_epoch_{}.ot", epoch));
            vs.save(&checkpoint_path)?;
            println!("    > Checkpoint saved to: {:?}", checkpoint_path);
        }
    }

    println!("\n--- Deep Training Complete ---");
    Ok(())
}
