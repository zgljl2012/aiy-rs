
use anyhow::Ok;
use diffusers::models::{vae, unet_2d};
use diffusers::transformers::clip::{Tokenizer, Config};
use diffusers::transformers::clip;
use tch::nn::Module;
use regex;
use tch::{Tensor, Device};

use crate::{bpe::Bpe, utils::get_device};

const PAT: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

/// AiyConfig 定义 Aiy 配置
pub struct AiyConfig {
    pub vocab_path: String,
    pub clip_weights_path: String,
    pub clip_config: Config,
    pub vae_weights_path: String,
    pub unet_weights_path: String,
    pub unet_config: unet_2d::UNet2DConditionModelConfig
}

pub struct AiyStableDiffusion {
    // clip
    pub clip_device: Device,
    clip_model: clip::ClipTextTransformer,
    // vae
    pub vae_device: Device,
    vae_model: vae::AutoEncoderKL,
    // unet
    pub unet_model: unet_2d::UNet2DConditionModel,
    pub unet_device: Device,
    // 分词器
    bpe: Bpe,
    tokenizer: Tokenizer
}

impl AiyStableDiffusion {
    pub fn new(cfg: AiyConfig) -> anyhow::Result<Self> {
        let clip_device = get_device();
        let unet_device = get_device();
        let vae_device = get_device();
        let bpe = Bpe::new(cfg.vocab_path)?;
        let clip_model = AiyStableDiffusion::build_clip_transformer(&cfg.clip_config, &cfg.clip_weights_path, clip_device)?;
        let tokenizer = AiyStableDiffusion::create_tokenizer(&bpe, cfg.clip_config.clone())?;
        // VAE
        let vae_model = AiyStableDiffusion::build_vae(&cfg.vae_weights_path, vae_device)?;
        // UNET
        let in_channels = 4;
        let unet_model = AiyStableDiffusion::build_unet(&cfg.unet_weights_path, unet_device, in_channels, cfg.unet_config)?;
        Ok(Self { tokenizer, bpe, clip_device, unet_device, clip_model, vae_model, vae_device, unet_model })
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

    fn build_vae(
        vae_weights: &str,
        device: Device,
    ) -> anyhow::Result<vae::AutoEncoderKL> {
        let mut vs_ae = tch::nn::VarStore::new(device);
        let autoencoder = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
        };
        // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json
        let autoencoder = vae::AutoEncoderKL::new(vs_ae.root(), 3, 3, autoencoder.clone());
        vs_ae.load(vae_weights)?;
        Ok(autoencoder)
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

    pub fn vae_decode(&self, t: &Tensor) -> Tensor {
        self.vae_model.decode(&(t / 0.18215))
    }

    fn build_unet(
        unet_weights: &str,
        device: Device,
        in_channels: i64,
        unet_cfg: unet_2d::UNet2DConditionModelConfig,
    ) -> anyhow::Result<unet_2d::UNet2DConditionModel> {
        let mut vs_unet = tch::nn::VarStore::new(device);
        let unet =
            unet_2d::UNet2DConditionModel::new(vs_unet.root(), in_channels, 4, unet_cfg);
        vs_unet.load(unet_weights)?;
        Ok(unet)
    }

}
