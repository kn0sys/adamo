// src/bin/process_data.rs

use adamo::text_processing::TextProcessor;
// ADDED: Imports for bincode 2.0
use bincode::{config, serde::encode_to_vec};
use std::env;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    println!("--- Adamo Full Dataset Processor ---");

    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: cargo run --bin process_data <tokenizer.json> <input_text_file> <output_file.bin>");
        return Err("Missing required arguments".into());
    }
    let tokenizer_path = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];

    println!("> Loading tokenizer from: {}", tokenizer_path);
    println!("> Reading dataset from:   {}", input_path);

    let processor = TextProcessor::new(tokenizer_path)?;
    let raw_text = fs::read_to_string(input_path)?;

    let articles: Vec<&str> = raw_text
        .split("\n = ")
        .filter(|s| !s.trim().is_empty() && s.len() > 200)
        .collect();

    let total_articles = articles.len();
    println!("\n> Found {} processable articles.", total_articles);
    println!("> Processing all articles...");

    let start_time = Instant::now();
    let mut processed_frames = Vec::new();
    let mut processed_count = 0;

    for (i, article) in articles.iter().enumerate() {
        if (i + 1) % 100 == 0 {
             print!("\r  - Progress: {}/{}", i + 1, total_articles);
             std::io::stdout().flush()?;
        }

        if let Ok(frame) = processor.text_to_frame(article) {
            processed_frames.push(frame);
            processed_count += 1;
        }
    }

    let duration = start_time.elapsed();
    println!("\n\n> Successfully processed {} / {} articles in {:?}", processed_count, total_articles, duration);

    // UPDATED: Use the bincode 2.0 API for serialization.
    println!("> Serializing and saving frames to {}...", output_path);
    let start_save_time = Instant::now();
    let config = config::standard();
    let serialized_data = encode_to_vec(&processed_frames, config)?;
    fs::write(output_path, serialized_data)?;
    let save_duration = start_save_time.elapsed();

    println!("> Training data saved successfully in {:?}.", save_duration);

    println!("\n--- Processing Complete ---");
    Ok(())
}
