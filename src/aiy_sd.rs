
use std::time::SystemTime;

use anyhow::Ok;
use crate::model_kind::ModelKind;
use crate::vae;
use diffusers::schedulers::PredictionType;
use diffusers::schedulers::ddim::{self, DDIMSchedulerConfig};
use crate::clip::{Tokenizer, Config};
use crate::clip;
use tch::nn::Module;
use regex;
use tch::{Tensor, Device, Kind};

use crate::utils::output_filename;
use crate::{bpe::Bpe, utils::get_device};
use crate::unet_2d;

const GUIDANCE_SCALE: f64 = 7.5;

const PAT: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

/// AiyConfig 定义 Aiy 配置
pub struct AiyConfig {
    pub vocab_path: String,
    pub clip_weights_path: String,
    pub clip_config: Config,
    pub vae_weights_path: String,
    pub vae_fp16: Option<bool>,
    pub unet_weights_path: String,
    pub unet_fp16: Option<bool>,
    pub unet_config: unet_2d::UNet2DConditionModelConfig,
    pub base_model: ModelKind
}

pub struct AiyStableDiffusion {
    // clip
    pub clip_device: Device,
    clip_model: clip::ClipTextTransformer,
    // vae
    pub vae_device: Device,
    vae_model: vae::AutoEncoderKL,
    vae_fp16: bool,
    // unet
    pub unet_model: unet_2d::UNet2DConditionModel,
    pub unet_device: Device,
    unet_fp16: bool,
    // 分词器
    bpe: Bpe,
    tokenizer: Tokenizer,
    // 基础模型
    pub base_model: ModelKind
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
        let vae_model = AiyStableDiffusion::build_vae(&cfg.vae_weights_path, vae_device, cfg.base_model.clone())?;
        // UNET
        let in_channels = 4;
        let unet_model = AiyStableDiffusion::build_unet(&cfg.unet_weights_path, unet_device, in_channels, cfg.unet_config)?;
        let unet_fp16 = cfg.unet_fp16.unwrap_or(false);
        let vae_fp16 = cfg.vae_fp16.unwrap_or(false);
        Ok(Self { tokenizer, bpe, clip_device, unet_device, clip_model, vae_model, vae_device, unet_model, unet_fp16, vae_fp16, base_model: cfg.base_model })
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
        base_model: ModelKind
    ) -> anyhow::Result<vae::AutoEncoderKL> {
        let mut vs_ae = tch::nn::VarStore::new(device);
        let autoencoder = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
        };
        // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json
        let autoencoder = vae::AutoEncoderKL::new(vs_ae.root(), 3, 3, autoencoder.clone(), base_model);
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

    fn build_scheduler(n_steps: usize, config: DDIMSchedulerConfig) -> ddim::DDIMScheduler {
        ddim::DDIMScheduler::new(n_steps, config)
    }

    pub fn run(&self, prompt: &str, negative_prompt: &str, final_image: &str, intermediary_images: bool, n_steps: usize, num_samples: i64, seed: i64, width: i64, height: i64, prediction_type: Option<PredictionType>) -> anyhow::Result<()> {
        let text_embeddings = self.embed_prompts(prompt, negative_prompt)?;
        // Scheduler
        let scheduler_config = ddim::DDIMSchedulerConfig { prediction_type: prediction_type.unwrap_or(PredictionType::Epsilon), ..Default::default() };
        let scheduler = AiyStableDiffusion::build_scheduler(n_steps, scheduler_config);

        let no_grad_guard = tch::no_grad_guard();
        let bsize = 1;
        let start = SystemTime::now();
        let kind = if self.unet_fp16 { Kind::Half } else { Kind::Float }; // Kind::Half
        for idx in 0..num_samples {
            tch::manual_seed(seed + idx);
            let mut latents = Tensor::randn(
                [bsize, 4, height / 8, width / 8],
                (kind, self.unet_device),
            );

            // scale the initial noise by the standard deviation required by the scheduler
            latents *= scheduler.init_noise_sigma();

            for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
                println!("Timestep {timestep_index}/{n_steps}");
                let latent_model_input = Tensor::cat(&[&latents, &latents], 0);
                let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep);
                let tm = text_embeddings.to_kind(kind);
                let noise_pred = self.unet_model.forward(&latent_model_input, timestep as f64, &tm);
                let noise_pred = noise_pred.chunk(2, 0);
                let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
                let noise_pred =
                    noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
                
                latents = scheduler.step(&noise_pred, timestep, &latents);
                // 生成中间过程图片
                if intermediary_images {
                    let latents = latents.to(self.vae_device);
                    let image = self.vae_decode(&latents);
                    let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
                    let image = (image * 255.).to_kind(Kind::Uint8);
                    let final_image =
                        output_filename(&final_image, idx + 1, num_samples, Some(timestep_index + 1));
                    tch::vision::image::save(&image, final_image)?;
                }
            }

            println!("Generating the final image for sample {}/{}.", idx + 1, num_samples);
            let mut latents = latents.to(self.vae_device);
            // 如果 unet 是 fp16 且 vae 不是 fp16
            if self.unet_fp16 && !self.vae_fp16 {
                latents = latents.to_kind(Kind::Float)
            } else if self.vae_fp16 && !self.unet_fp16 {
                latents = latents.to_kind(Kind::Half)
            }
            let image = self.vae_decode(&latents);
            let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
            let image = (image * 255.).to_kind(Kind::Uint8);
            let final_image = output_filename(&final_image, idx + 1, num_samples, None);
            tch::vision::image::save(&image, final_image)?;
        }
        println!("=== Generated image: {:?}", SystemTime::now().duration_since(start).unwrap());
        drop(no_grad_guard);
        Ok(())
    }

}
