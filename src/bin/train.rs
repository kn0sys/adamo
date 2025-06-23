// src/bin/train.rs

use adamo::{
    generative_model::GenerativeModel,
    text_processing::TextProcessor,
};
use anyhow::{anyhow, Result};
use arrow::array::StringArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use regex::Regex;
use std::{env, fs::{self, File}, path::Path, time::Instant};
use tch::{nn, nn::{OptimizerConfig, ModuleT}, Device, Tensor};
use std::f64::consts::PI;

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
        "raw" | "txt" => { /* unchanged */ Ok(fs::read_to_string(file_path)?.split("\n \n").map(|s| s.to_string()).collect()) }
        "parquet" => { /* unchanged */
            let file = File::open(file_path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let text_column_name = "text";
            if builder.schema().field_with_name(text_column_name).is_err() { anyhow::bail!("Parquet file must contain a column named '{}'", text_column_name); }
            let reader = builder.build()?;
            let mut articles = Vec::new();
            for record_batch in reader {
                let record_batch = record_batch?;
                let text_column = record_batch.column_by_name(text_column_name).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
                for text_val in text_column.iter() { if let Some(text) = text_val { articles.push(text.to_string()); } }
            }
            Ok(articles)
        }
        _ => { anyhow::bail!("Unsupported file type: '{}'. Please use .raw, .txt, or .parquet", extension); }
    }
}

fn text_to_batch(
    processor: &TextProcessor,
    text: &str,
    device: Device,
) -> Result<Option<(Tensor, Tensor)>> {
    let encoding = processor.encode(text)?;
    let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    if token_ids.len() < 2 { return Ok(None); }
    let input_ids = &token_ids[..token_ids.len() - 1];
    let target_ids = &token_ids[1..];
    let xs = Tensor::f_from_slice(input_ids)?.to(device).view((1, -1));
    let ys = Tensor::f_from_slice(target_ids)?.to(device).view([-1]);
    Ok(Some((xs, ys)))
}

fn main() -> Result<()> {
    println!("--- Adamo CPU-Optimized Training ---");
    let device = Device::Cpu;

    // --- 1. Setup ---
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: cargo run --bin train <tokenizer.json> <output_model.ot> <file1> [file2]...");
        return Err(anyhow!("Missing required arguments"));
    }
    let tokenizer_path = &args[1];
    let output_model_path = &args[2];
    let training_data_paths: Vec<&String> = args.iter().skip(3).collect();

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
    // Let's use a slightly smaller model better suited for CPU training
    let d_model = 256;
    let nhead = 4;
    let num_layers = 4;
    let dim_feedforward = 1024;
    let generative_model = GenerativeModel::new(&vs, vocab_size, d_model, nhead, num_layers, dim_feedforward);

    let initial_lr = 1e-4;
    let mut optimizer = nn::Adam::default().build(&vs, initial_lr)?;

    println!("> Adamo Transformer Initialized for CPU optimization.");

    // --- 3. Training Loop with Optimizations ---
    println!("\n> Starting deep training loop...");
    let num_epochs = 20;
    // --- NEW: Gradient Accumulation Setup ---
    let accumulation_steps = 8; // Simulate a batch size 8 times larger
    let clip_grad_norm = 1.0;   // Gradient clipping value

    let total_training_steps = num_epochs * (all_articles.len() / accumulation_steps);
    let mut current_step = 0;

    for epoch in 1..=num_epochs {
        let epoch_start_time = Instant::now();
        let mut total_loss = 0.0;
        let mut article_idx = 0;

        optimizer.zero_grad(); // Zero the gradients at the start of the epoch

        let mut final_lr = 0.0;
        for article in &all_articles {
            let cleaned_article = clean_text(article);
            if let Some((xs, ys)) = text_to_batch(&processor, &cleaned_article, device)? {
                let logits = generative_model.forward_t(&xs, true);
                let loss = logits.view([-1, vocab_size]).cross_entropy_for_logits(&ys);

                // Accumulate loss for a more stable average
                total_loss += loss.double_value(&[]);

                // Scale the loss by accumulation steps and backpropagate
                (loss / accumulation_steps as f64).backward();

                article_idx += 1;
                // --- NEW: Optimizer Step after accumulating gradients ---
                if article_idx % accumulation_steps == 0 {
                    // Clip gradients to prevent explosions
                    optimizer.clip_grad_norm(clip_grad_norm);
                    optimizer.step();
                    optimizer.zero_grad(); // Zero grads after stepping

                    // Update learning rate (we step on accumulated batches)
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
        }
    }

    vs.save(output_model_path)?;
    println!("\n> Final model weights saved to {}", output_model_path);
    println!("\n--- Deep Training Complete ---");
    Ok(())
}
