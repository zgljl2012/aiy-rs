use std::fs;

use super::Activation;
use anyhow::Ok;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: i64,
    pub embed_dim: i64,         // aka config.hidden_size
    pub activation: Activation, // aka config.hidden_act
    pub intermediate_size: i64,
    pub max_position_embeddings: usize,
    // The character to use for padding, use EOS when not set.
    pub pad_with: Option<String>,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    #[allow(dead_code)]
    pub projection_dim: i64,
}

impl Config {

    pub fn from_file<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<Self> {
        let file = fs::read_to_string(path)?;
        let cfg: Config = toml::from_str(&file)?;
        Ok(cfg)
    }

    // The config details can be found in the "text_config" section of this json file:
    // https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
    pub fn v1_5() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 768,
            activation: Activation::QuickGelu,
        }
    }

    // https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/text_encoder/config.json
    pub fn v2_1() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1024,
            intermediate_size: 4096,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 23,
            num_attention_heads: 16,
            projection_dim: 512,
            activation: Activation::Gelu,
        }
    }

    // https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9/blob/main/text_encoder/config.json
    pub fn sdxl_v_0_9() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 768,
            activation: Activation::QuickGelu,
        }
    }

    pub fn sdxl_v_0_9_encoder2() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1280,
            intermediate_size: 5120,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 32,
            num_attention_heads: 20,
            projection_dim: 1280,
            activation: Activation::Gelu,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_load_from_toml() {
        let cfg = Config::from_file("src/clip/config.default.toml").unwrap();
        assert_eq!(cfg.activation, Activation::QuickGelu);
        assert_eq!(cfg.embed_dim, 768);
        assert_eq!(cfg.vocab_size, 49408);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.max_position_embeddings, 77);
        assert_eq!(cfg.pad_with, Some("!".to_string()));
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.projection_dim, 768);
        println!("{:?}", cfg);
    }
}
