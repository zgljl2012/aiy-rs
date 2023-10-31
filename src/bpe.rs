use std::{collections::HashMap, io::BufRead};

use crate::{bytes_to_unicode::BYTES_TO_UNICODE, utils};
use anyhow::Ok;

pub struct Bpe {
    pub encoder: HashMap<String, usize>,
    pub decoder: HashMap<usize, String>,
    pub bpe_ranks: HashMap<(String, String), usize>,
    pub start_of_text_token: usize,
    pub end_of_text_token: usize,
}

impl Bpe {
    pub fn new(bpe_path: String) -> anyhow::Result<Self> {
        let bpe_file = utils::file_open(bpe_path)?;
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
        let bpe_ranks: HashMap<_, _> = bpe_lines
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        Ok(Self {
            encoder,
            decoder,
            bpe_ranks,
            start_of_text_token,
            end_of_text_token,
        })
    }
}
