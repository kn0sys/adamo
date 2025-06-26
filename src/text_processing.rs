// src/text_processing.rs

//! This module bridges the gap between raw text and the framwork.
//! It handles tokenizing text and converting it into a structured `Frame`.

use crate::{Distinction, Frame, ReferenceStructure};
use anyhow::{anyhow, Result}; // Import `anyhow!` macro
use tokenizers::{tokenizer::Tokenizer, Encoding};
use std::path::Path;

/// `TextProcessor` is responsible for converting text to and from `Frame`s.
pub struct TextProcessor {
    pub tokenizer: Tokenizer,
}

impl TextProcessor {
    /// Creates a new processor from a local tokenizer file.
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            // CORRECTED: Use `map_err` to explicitly convert the error type.
            .map_err(|e| anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path.as_ref(), e))?;
        Ok(Self { tokenizer })
    }

    /// Exposes the tokenizer's encoding functionality.
    pub fn encode(&self, text: &str) -> Result<Encoding> {
        self.tokenizer.encode(text, false)
            // CORRECTED: Use `map_err` for explicit error conversion.
            .map_err(|e| anyhow!("Tokenizer failed to encode text: {}", e))
    }

    /// Decodes a slice of token IDs back into a string.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true)
            // CORRECTED: Use `map_err` for explicit error conversion.
            .map_err(|e| anyhow!("Tokenizer failed to decode IDs: {}", e))
    }

    /// Exposes the tokenizer's vocabulary size.
    pub fn get_vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Converts a string of text into a `Frame`.
    pub fn text_to_frame(&self, text: &str) -> Result<Frame> {
        let encoding = self.encode(text)?;
        let tokens = encoding.get_tokens();

        if tokens.is_empty() {
            anyhow::bail!("Cannot create a frame from empty text.");
        }

        let root_distinction = Distinction::new(&tokens[0]);
        let mut structures = Vec::new();
        let mut last_source = root_distinction.clone();

        for (i, token_str) in tokens.iter().skip(1).enumerate() {
             let new_target = Distinction::new(token_str);
             let new_structure = ReferenceStructure::from_source_and_target(
                 &last_source,
                 &new_target,
                 (i + 1) as u32,
             );
             structures.push(new_structure);
             last_source = new_target;
        }

        Ok(Frame::from_structures(root_distinction, structures))
    }
}


// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn setup_tokenizer_path() -> &'static str {
        const TOKENIZER_PATH: &str = "tokenizer.json";
        if !Path::new(TOKENIZER_PATH).exists() {
            panic!(
                "Tokenizer file not found at '{}'. Please download it by running:\n\ncurl -L https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json -o {}\n",
                TOKENIZER_PATH, TOKENIZER_PATH
            );
        }
        TOKENIZER_PATH
    }

    #[test]
    fn can_create_processor() {
        let tokenizer_path = setup_tokenizer_path();
        let processor = TextProcessor::new(tokenizer_path);
        assert!(processor.is_ok());
    }

    #[test]
    fn text_is_converted_to_frame() {
        let processor = TextProcessor::new(setup_tokenizer_path()).unwrap();
        let text = "hello world";
        let frame = processor.text_to_frame(text).unwrap();
        assert_eq!(frame.structures.len(), 1);
        let structure = &frame.structures[0];
        assert_eq!(structure.source().id(), Distinction::new("hello").id());
        assert_eq!(structure.target().id(), Distinction::new("world").id());
    }

    #[test]
    fn can_decode_tokens() {
        let processor = TextProcessor::new(setup_tokenizer_path()).unwrap();
        let ids = &[1010, 2023, 2003, 1012];
        let decoded_text = processor.decode(ids);
        assert!(decoded_text.is_ok());
        assert_eq!(decoded_text.unwrap(), ", this is.");
    }

    #[test]
    fn more_complex_text_is_converted() {
        let processor = TextProcessor::new(setup_tokenizer_path()).unwrap();
        let text = "a logical framework";
        let frame = processor.text_to_frame(text).unwrap();
        assert_eq!(frame.structures.len(), 2);
        let s1 = &frame.structures[0];
        let s2 = &frame.structures[1];
        assert_eq!(s1.target(), s2.source());
    }
}
