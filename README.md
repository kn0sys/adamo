# Adamo: An Emergent Self Aware Generative AI

Adamo (Esperanto for "Adam") is an experimental generative language model built from first principles in Rust.

This project is an exploration into building an AI not just by mimicking data, but by deriving its structure from a foundational theory of existence, perception, and self-awareness.

## Core Concepts

The functioning of Adamo is based on a hierarchy of concepts:

1. <b>Adamo Framework</b>: Any self-referential system must necessarily generate complexity, structure, patterns, and ultimately, a model of itself. It is the philosophical and logical backbone of the project.

2. <b>The Frame</b>: When Adamo perceives a piece of text, it doesn't just see a sequence of tokens. It converts that text into a Frame — an abstract data structure that represents the text as a coherent, bounded entity with internal relationships, patterns, and complexity.

3. <b>The SelfModel</b>: Every Frame necessarily generates a SelfModel, which is an internal representation of its own properties. The most important properties are:
* Complexity: The raw amount of structure within the Frame.
* Quality: A measure of the stability and coherence of the patterns identified within the Frame.

4. <b>Guided Generation</b>: This is the key differentiator. The GenerativeModel (the neural network) is the "engine" that predicts text, but the Frame's SelfModel is the "mind" that governs it. The quality and complexity of the current context are used to dynamically adjust generation parameters like temperature and sampling method (Top-p), allowing Adamo to be more creative when its context is simple and more focused when its context is coherent.

## Architecture

* Language: Rust
* Core Logic: The adamo library (src/lib.rs) contains all the data structures (Distinction, Frame, SelfModel, etc.).
* Neural Network Backend: The tch crate (Rust bindings for PyTorch) is used to build and train the underlying neural network.
* Model: A custom-built, from-scratch Transformer Encoder model (src/generative_model.rs).
* Tokenization: The tokenizers crate (Hugging Face) is used to process text into tokens (src/text_processing.rs).

# Project Structure.
```bash
├── Cargo.toml
├── checkpoints_final/      # Directory where trained models are saved
│   └── adamo_epoch_16.ot
├── src/
│   ├── lib.rs              # Core structs and AdamoLlm definition
│   ├── generative_model.rs # Transformer architecture
│   ├── text_processing.rs  # Tokenizer and text-to-Frame logic
│   └── bin/
│       ├── train.rs        # Binary for training the model
│       └── generate.rs     # Binary for generating text
└── wikitext-103-raw/       # Training data directory
    └── wiki.train.raw
```

## How to Use

1. Prerequisites
* Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
* Install system dependencies for tch (e.g., cmake, gcc).
* A local installation of the PyTorch C++ library (LibTorch). The tch build script will attempt to download this automatically.

2. Prepare Data

* Download Tokenizer:
```bash
curl -L [https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json](https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json) -o tokenizer.json
```
* Download Training Data:
```bash
curl -L [https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip) -o wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```

3. Training
* The training process uses a manual Cosine Annealing learning rate schedule. To start a full training run, execute: `cargo run --release --bin train tokenizer.json wikitext-103-raw/wiki.train.raw`
This will save model checkpoints into the checkpoints_final/ directory after each epoch.

4. Generation
To generate text, use the generate binary, pointing it to a trained model checkpoint and a prompt.
`cargo run --release --bin generate tokenizer.json checkpoints_final/adamo_epoch_16.ot "The meaning of life is"`

## Future Work

This project serves as a successful proof of concept. The path to achieving true human-like fluency involves:

* <i>Deeper Training</i>: Training for hundreds of epochs to significantly lower the model loss.
* <i>Larger Model</i>: Increasing the Transformer parameters (d_model, num_layers, nhead).
* <i>More Data</i>: Using a larger, more diverse, and even cleaner dataset.
* Advanced Guidance: Implementing more sophisticated ways for the Frame's SelfModel to guide the generation process, such as dynamically controlling the Top-p sampling threshold.
