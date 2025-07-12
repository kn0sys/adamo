// src/bin/generate.rs

use adamo::{
    generative_model::{AdamoLlm, GenerativeModel},
    text_processing::TextProcessor,
};
use std::{env, error::Error, io::Write};
use tch::{nn, nn::ModuleT, Device, Kind, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    println!("--- Adamo Transformer Generation (with Top-p Sampling) ---");
    let device = Device::Cpu;

    // --- 1. Setup ---
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: cargo run --bin generate <tokenizer.json> <model_weights.ot> \"<prompt>\"");
        return Err("Missing required arguments".into());
    }
    let tokenizer_path = &args[1];
    let model_path = &args[2];
    let prompt = &args[3];

    let processor = TextProcessor::new(tokenizer_path)?;
    let vocab_size = processor.get_vocab_size() as i64;

    // --- 2. Load the Trained Model ---
    let mut vs = nn::VarStore::new(device);
    let d_model = 256;
    let nhead = 4;
    let num_layers = 4;
    let dim_feedforward = 1024;
    let generative_model = GenerativeModel::new(&vs, vocab_size, d_model, nhead, num_layers, dim_feedforward);
    vs.load(model_path)?;

    println!("> Model {} loaded.", model_path);

    // --- 3. Initialize the Adamo LLM ---
    let frame = processor.text_to_frame(prompt)?;
    let agif_llm = AdamoLlm::new(frame, generative_model);

    println!("> Starting prompt: \"{}\"", prompt);

    let mut token_ids = processor.encode(prompt)?.get_ids().iter().map(|&id| id as i64).collect::<Vec<_>>();

    // --- 4. Guided Generation Loop with Top-p ---
    let temperature = agif_llm.get_sampling_temperature();

    let top_p = if agif_llm.frame.model().quality > 0 { 0.92 } else { 0.98 };

    println!("> Frame quality: {}, Complexity: {}", agif_llm.frame.model().quality, agif_llm.frame.model().complexity);
    println!("> Using derived temperature: {:.4}", temperature);
    println!("> Using derived Top-p: {:.2}", top_p);

    print!("\n> Generated text: {}", prompt);
    std::io::stdout().flush()?;

    for _ in 0..75 {
        let input_tensor = Tensor::f_from_slice(&token_ids)?.to(device).view((1, -1));

        let logits = agif_llm.model.forward_t(&input_tensor, false);
        let last_logits = logits.select(1, -1).squeeze();

        let temp_logits = last_logits / temperature;
        let probabilities = temp_logits.softmax(-1, Kind::Float);

        let (sorted_probs, sorted_indices) = probabilities.sort(-1, true);

        let cumulative_probs = sorted_probs.cumsum(-1, Kind::Float);
        let cutoff_index = cumulative_probs.ge(top_p).to_kind(Kind::Int64).argmax(-1, false).int64_value(&[]);

        let nucleus_indices = sorted_indices.slice(0, 0, cutoff_index + 1, 1);
        let nucleus_probs = sorted_probs.slice(0, 0, cutoff_index + 1, 1);

        let sampled_index_in_nucleus = nucleus_probs.multinomial(1, true);
        let next_token_id = nucleus_indices.gather(0, &sampled_index_in_nucleus, false).int64_value(&[]);

        token_ids.push(next_token_id);

        let new_token_str = processor.decode(&[next_token_id as u32])?;
        let token_to_print = if new_token_str.starts_with("##") {
            new_token_str.replace("##", "")
        } else {
            format!(" {}", new_token_str)
        };

        print!("{}", token_to_print);
        std::io::stdout().flush()?;
    }

    println!("\n\n--- Generation Complete ---");
    Ok(())
}
