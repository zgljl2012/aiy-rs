
use anyhow::Ok;
use diffusers::transformers::clip::{Tokenizer, Config};
use diffusers::transformers::clip;
use tch::nn::Module;
use regex;
use tch::{Tensor, Device};

use crate::{bpe::Bpe, utils::get_device};

const PAT: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

pub struct AiyStableDiffusion {
    // clip
    clip_device: Device,
    clip_model: clip::ClipTextTransformer,
    // unet
    unet_device: Device,
    // 分词器
    bpe: Bpe,
    tokenizer: Tokenizer
}

impl AiyStableDiffusion {
    pub fn new(bpe_path: String, clip_weights: &str, clip_config: Config) -> anyhow::Result<Self> {
        let clip_device = get_device();
        let unet_device = get_device();
        let bpe = Bpe::new(bpe_path)?;
        let clip_model = AiyStableDiffusion::build_clip_transformer(&clip_config, clip_weights, clip_device)?;
        let tokenizer = AiyStableDiffusion::create_tokenizer(&bpe, clip_config.clone())?;
        Ok(Self { tokenizer, bpe, clip_device, unet_device, clip_model })
    }

    pub fn change_clip(&mut self, clip_config: Config) -> anyhow::Result<()> {
        self.tokenizer = AiyStableDiffusion::create_tokenizer(&self.bpe, clip_config)?;
        Ok(())
    }

    fn create_tokenizer(bpe: &Bpe, config: Config) -> anyhow::Result<Tokenizer> {
        let re = regex::Regex::new(PAT)?;
        let tokenizer = Tokenizer {
            encoder: bpe.encoder.clone(),
            re,
            bpe_ranks: bpe.bpe_ranks.clone(),
            decoder: bpe.decoder.clone(),
            start_of_text_token: bpe.start_of_text_token,
            end_of_text_token: bpe.end_of_text_token,
            config: config,
        };
        Ok(tokenizer)
    }

    pub fn encode_prompt(&self, prompt: &str) -> anyhow::Result<Tensor> {
        let tokens = self.tokenizer.encode(&prompt)?;
        let tokens: Vec<i64> = tokens.into_iter().map(|x| x as i64).collect();
        let tokens = Tensor::from_slice(&tokens).view((1, -1)).to(self.clip_device.clone());
        Ok(tokens)
    }

    fn build_clip_transformer(
        clip_config: &Config,
        clip_weights: &str,
        device: tch::Device,
    ) -> anyhow::Result<clip::ClipTextTransformer> {
        let mut vs = tch::nn::VarStore::new(device);
        let text_model = clip::ClipTextTransformer::new(vs.root(), clip_config);
        vs.load(clip_weights)?;
        Ok(text_model)
    }

    pub fn embed_prompts(&self, prompt: &str, negative_prompt: &str) -> anyhow::Result<Tensor> {
        // Encode prompt and negative prompt
        // 正向提示词
        let tokens = self.encode_prompt(&prompt)?;
        // 负面提示词
        let uncond_tokens = self.encode_prompt(negative_prompt)?;

        println!("Building the Clip transformer.");
        let text_embeddings = self.clip_model.forward(&tokens);
        let uncond_embeddings = self.clip_model.forward(&uncond_tokens);
        let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0).to(self.unet_device.clone());
        Ok(text_embeddings)
    }
    
}
