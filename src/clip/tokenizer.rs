use std::{collections::{HashMap, HashSet}, io::BufRead};

use anyhow::Ok;
use tch::{Tensor, Device};

use super::{Config, bytes_to_unicode::BYTES_TO_UNICODE};

const PAT: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

// This is mostly a Rust rewrite of the original Python CLIP code.
// https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
/// A tokenizer for CLIP.
pub struct Tokenizer {
    pub re: regex::Regex,
    pub encoder: HashMap<String, usize>,
    pub decoder: HashMap<usize, String>,
    pub bpe_ranks: HashMap<(String, String), usize>,
    pub start_of_text_token: usize,
    pub end_of_text_token: usize,
    pub config: Config,
    pub device: Device
}

impl Tokenizer {
    /// Creates a new CLIP tokenizer, this takes as input the path for the bpe vocabulary file.
    pub fn create<T: AsRef<std::path::Path> + std::fmt::Debug>(
        bpe_path: T,
        device: Device,
        c: &Config,
    ) -> anyhow::Result<Tokenizer> {
        let bpe_file = crate::utils::file_open(bpe_path)?;
        let bpe_lines: Result<Vec<String>, _> = std::io::BufReader::new(bpe_file).lines().collect();
        let bpe_lines = bpe_lines?;
        let bpe_lines: Result<Vec<_>, _> = bpe_lines[1..49152 - 256 - 2 + 1]
            .iter()
            .map(|line| {
                let vs: Vec<_> = line.split_whitespace().collect();
                if vs.len() != 2 {
                    anyhow::bail!("expected two items got {} '{}'", vs.len(), line)
                }
                Ok((vs[0].to_string(), vs[1].to_string()))
            })
            .collect();
        let bpe_lines = bpe_lines?;
        let mut vocab: Vec<String> = Vec::new();
        for (_index, elem) in BYTES_TO_UNICODE {
            vocab.push(elem.into())
        }
        for (_index, elem) in BYTES_TO_UNICODE {
            vocab.push(format!("{elem}</w>"));
        }
        for elem in bpe_lines.iter() {
            vocab.push(format!("{}{}", elem.0, elem.1))
        }
        let start_of_text_token = vocab.len();
        vocab.push("<|startoftext|>".to_string());
        let end_of_text_token = vocab.len();
        vocab.push("<|endoftext|>".to_string());
        let encoder: HashMap<_, _> = vocab.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let decoder: HashMap<_, _> = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();
        let bpe_ranks: HashMap<_, _> =
            bpe_lines.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let re = regex::Regex::new(PAT)?;
        let tokenizer = Tokenizer {
            encoder,
            re,
            bpe_ranks,
            decoder,
            start_of_text_token,
            end_of_text_token,
            config: c.clone(),
            device
        };
        Ok(tokenizer)
    }



    fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
        let mut pairs = HashSet::new();
        for (i, v) in word.iter().enumerate() {
            if i > 0 {
                pairs.insert((word[i - 1].clone(), v.clone()));
            }
        }
        pairs
    }

    fn bpe(&self, token: &str) -> Vec<usize> {
        let mut word: Vec<String> = token.chars().map(|x| x.to_string()).collect();
        if word.is_empty() {
            return Vec::new();
        }
        let last_index = word.len() - 1;
        word[last_index] = format!("{}</w>", word[last_index]);
        while word.len() > 1 {
            let mut current_min = None;
            let pairs = Self::get_pairs(&word);
            for p in pairs.iter() {
                match self.bpe_ranks.get(p) {
                    None => {}
                    Some(v) => {
                        let should_replace = match current_min {
                            None => true,
                            Some((current_min, _)) => v < current_min,
                        };
                        if should_replace {
                            current_min = Some((v, p))
                        }
                    }
                }
            }
            let (first, second) = match current_min {
                None => break,
                Some((_v, (first, second))) => (first, second),
            };
            let mut new_word = vec![];
            let mut index = 0;
            while index < word.len() {
                let w = &word[index];
                if index + 1 < word.len() && w == first && &word[index + 1] == second {
                    new_word.push(format!("{first}{second}"));
                    index += 2
                } else {
                    new_word.push(w.clone());
                    index += 1
                }
            }
            word = new_word
        }
        word.iter().filter_map(|x| self.encoder.get(x)).copied().collect()
    }

    pub fn encode_pad(&self, s: &str, pad_size_to: Option<usize>) -> anyhow::Result<Vec<usize>> {
        let s = s.to_lowercase();
        let mut bpe_tokens: Vec<usize> = vec![self.start_of_text_token];
        for token in self.re.captures_iter(&s) {
            let token = token.get(0).unwrap().as_str();
            bpe_tokens.extend(self.bpe(token))
        }
        match pad_size_to {
            None => bpe_tokens.push(self.end_of_text_token),
            Some(pad_size_to) => {
                bpe_tokens.push(self.end_of_text_token);
                bpe_tokens.resize_with(
                    std::cmp::min(bpe_tokens.len(), pad_size_to - 1),
                    Default::default,
                );
                let pad_with = match &self.config.pad_with {
                    None => self.end_of_text_token,
                    Some(pad_with) => match self.encoder.get(pad_with) {
                        None => anyhow::bail!("no encoding for padding character {}", pad_with),
                        Some(v) => *v,
                    },
                };
                while bpe_tokens.len() < pad_size_to {
                    bpe_tokens.push(pad_with)
                }
            }
        }
        Ok(bpe_tokens)
    }

    /// The main tokenization entry point, takes as input a string and returns the list of tokens.
    pub fn encode(&self, s: &str) -> anyhow::Result<Vec<usize>> {
        self.encode_pad(s, Some(self.config.max_position_embeddings))
    }

    /// The inverse of the tokenization process, takes as input a list of tokens and returns a
    /// string that produces this tokenization.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let s: String = tokens.iter().map(|token| self.decoder[token].as_str()).collect();
        s.replace("</w>", " ")
    }

    pub fn parse_prompt(&self, prompt: &str) -> anyhow::Result<Tensor> {
        let tokens = self.encode(&prompt)?;
        let tokens: Vec<i64> = tokens.into_iter().map(|x| x as i64).collect();
        let tokens = Tensor::from_slice(&tokens)
            .view((1, -1))
            .to(self.device.clone());
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use crate::{model_kind::ModelKind, clip::bpe::Bpe, aiy_sd::AiyStableDiffusion};

    #[test]
    fn test_tokenizer() {
        let bpe_path = "data/bpe_simple_vocab_16e6.txt";
        let bpe = Bpe::new(bpe_path.to_string()).unwrap();
        let tokenizer = AiyStableDiffusion::create_tokenizer(&bpe, tch::Device::Cpu, ModelKind::SD2_1.clip_config()).unwrap();
        let r = tokenizer.encode("A horse with a dog").unwrap();
        println!("{:?}", r)
    }
}