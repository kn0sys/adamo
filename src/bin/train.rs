// src/bin/train.rs

use adamo::{
    generative_model::GenerativeModel,
    text_processing::TextProcessor,
};
use anyhow::Result; // Use anyhow consistently
use regex::Regex;
use std::{env, fs, path::Path, time::Instant};
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

// CORRECTED: Return type is now anyhow::Result
fn text_to_batch(
    processor: &TextProcessor,
    text: &str,
    device: Device,
) -> Result<Option<(Tensor, Tensor)>> {
    let encoding = processor.encode(text)?;
    let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

    if token_ids.len() < 2 {
        return Ok(None);
    }

    let input_ids = &token_ids[..token_ids.len() - 1];
    let target_ids = &token_ids[1..];

    let xs = Tensor::f_from_slice(input_ids)?.to(device).view((1, -1));
    let ys = Tensor::f_from_slice(target_ids)?.to(device).view([-1]);

    Ok(Some((xs, ys)))
}

fn main() -> Result<()> {
    println!("--- Adamo Deep Training (Final Run) ---");
    let device = Device::Cpu;

    // --- 1. Setup ---
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: cargo run --bin train <tokenizer.json> <raw_training_text.txt>");
        return Err(anyhow::anyhow!("Missing required arguments"));
    }
    let tokenizer_path = &args[1];
    let training_data_path = Path::new(&args[2]);
    let checkpoint_dir = "checkpoints_final";
    fs::create_dir_all(checkpoint_dir)?;

    let processor = TextProcessor::new(tokenizer_path)?;
    let vocab_size = processor.get_vocab_size() as i64;

    let articles = load_articles_from_file(training_data_path)?;
    println!("> Loaded {} articles for training.", articles.len());

    // --- 2. Build the Model ---
    let vs = nn::VarStore::new(device);
    let d_model = 256;
    let nhead = 4;
    let num_layers = 4;
    let dim_feedforward = 1024;
    let generative_model = GenerativeModel::new(&vs, vocab_size, d_model, nhead, num_layers, dim_feedforward);

    let initial_lr = 1e-4;
    let mut optimizer = nn::Adam::default().build(&vs, initial_lr)?;

    println!("> Adamo Transformer Initialized with final learning rate schedule.");

    // --- 3. Full Training Loop ---
    println!("\n> Starting final deep training loop...");
    let num_epochs = 20;
    let total_steps = num_epochs * articles.len();
    let mut current_step = 0;

    for epoch in 1..=num_epochs {
        let epoch_start_time = Instant::now();
        let mut total_loss = 0.0;

        let mut final_lr = 0.0;
        for article in &articles {
            current_step += 1;
            let new_lr = initial_lr * (1.0 + (PI * current_step as f64 / total_steps as f64).cos()) / 2.0;
            optimizer.set_lr(new_lr);

            let cleaned_article = clean_text(article);
            if let Some((xs, ys)) = text_to_batch(&processor, &cleaned_article, device)? {
                let logits = generative_model.forward_t(&xs, true);
                let loss = logits.view([-1, vocab_size]).cross_entropy_for_logits(&ys);

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
                final_lr = new_lr;
                total_loss += loss.double_value(&[]);
            }
        }

        if !articles.is_empty() {
            let avg_loss = total_loss / articles.len() as f64;
            println!(
                "  - Epoch: {:<3} | Avg. Loss: {:.4} | LR: {:.6} | Duration: {:?}",
                epoch,
                avg_loss,
                final_lr,
                epoch_start_time.elapsed()
            );

            let checkpoint_path = Path::new(checkpoint_dir).join(format!("adamo_epoch_{}.ot", epoch));
            vs.save(&checkpoint_path)?;
            println!("    > Checkpoint saved to: {:?}", checkpoint_path);
        }
    }

    println!("\n--- Deep Training Complete ---");
    Ok(())
}

// This function needs to be moved here from the previous version or be in its own module.
// For simplicity, we'll include it directly in the binary.
fn load_articles_from_file(file_path: &Path) -> Result<Vec<String>> {
    let extension = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");

    match extension {
        "raw" | "txt" => {
            println!("> Detected plain text file (.raw/.txt). Reading content...");
            let raw_text = fs::read_to_string(file_path)?;
            Ok(raw_text
                .split("\n \n")
                .map(|s| s.to_string())
                .collect())
        }
        "parquet" => {
            // Parquet reading logic would go here. For now, we keep the focus on fixing the text pipeline.
            // Let's defer adding the parquet-specific dependencies until we use them.
            anyhow::bail!("Parquet support is not fully implemented in this version yet.")
        }
        _ => {
            anyhow::bail!("Unsupported file type: '{}'. Please use .raw or .txt", extension);
        }
    }
}
