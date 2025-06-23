// src/text_processing.rs

//! This module bridges the gap between raw text and the model.
//! It handles tokenizing text and converting it into a structured `Frame`.

use crate::{Distinction, Frame, ReferenceStructure};
use tokenizers::{tokenizer::Tokenizer, Encoding};
use std::path::Path;

/// `TextProcessor` is responsible for converting text to and from`Frame`s.
pub struct TextProcessor {
    pub tokenizer: Tokenizer,
}

impl TextProcessor {
    /// Creates a new processor from a local tokenizer file.
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Box::<dyn std::error::Error>::from(format!("Failed to load tokenizer: {}", e)))?;
        Ok(Self { tokenizer })
    }

    /// Exposes the tokenizer's encoding functionality.
    pub fn encode(&self, text: &str) -> Result<Encoding, Box<dyn std::error::Error>> {
        self.tokenizer.encode(text, false)
            .map_err(|e| e.to_string().into())
    }

    // CORRECTED: Added the missing public decode method.
    /// Decodes a slice of token IDs back into a string.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
        self.tokenizer.decode(ids, true)
            .map_err(|e| e.to_string().into())
    }

    /// Exposes the tokenizer's vocabulary size.
    pub fn get_vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Converts a string of text into a `Frame`.
    pub fn text_to_frame(&self, text: &str) -> Result<Frame, Box<dyn std::error::Error>> {
        let encoding = self.encode(text)?;
        let tokens = encoding.get_tokens();

        if tokens.is_empty() {
            return Err("Cannot create a frame from empty text.".into());
        }

        let root_distinction = Distinction::new(&tokens[0]);
        let mut structures = Vec::new();
        let mut last_source = root_distinction.clone();

        for (i, token_str) in tokens.iter().skip(1).enumerate() {
             let new_target = Distinction::new(token_str);
             let new_structure = ReferenceStructure::from_source_and_target(
                 &last_source,
                 &new_target,
                 (i + 1) as u32
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
        let ids = &[1010, 2023, 2003, 1012]; // " ,  there is ."
        let decoded_text = processor.decode(ids);
        assert!(decoded_text.is_ok());
        assert_eq!(decoded_text.unwrap(), ", this is.");
    }
}
