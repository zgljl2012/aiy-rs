use std::time::SystemTime;

use crate::clip;
use crate::clip::{bpe::Bpe, Config, Tokenizer};
use crate::model_kind::ModelKind;
use crate::schedulers::ddim::DDIMSchedulerConfig;
use crate::schedulers::SchedulerKind;
use crate::vae;
use anyhow::Ok;
use regex;
use tch::nn::Module;
use tch::{Device, Kind, Tensor};

use crate::unet::unet_2d;
use crate::utils::output_filename;
use crate::utils::{get_device, image_preprocess};

mod builder;
pub mod config;
mod types;

pub use self::builder::AiySdBuilder;

const GUIDANCE_SCALE: f64 = 7.5;

const PAT: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

/// AiyConfig 定义 Aiy 配置
pub struct AiyConfig {
    pub vocab_path: String,
    pub clip_weights_path: String,
    pub vae_weights_path: String,
    pub vae_fp16: Option<bool>,
    pub unet_weights_path: String,
    pub unet_fp16: Option<bool>,
    pub base_model: ModelKind,
    pub width: usize,
    pub height: usize,
}

struct EmbededPrompts {
    text_embeddings: Tensor,
    pooled_prompt_embeds: Option<Tensor>,
    negative_pooled_prompt_embeds: Option<Tensor>,
}

pub struct AiyStableDiffusion {
    // clip
    pub clip_device: Device,
    pub(crate) clip_model: clip::ClipTextTransformer,
    // vae
    pub vae_device: Device,
    pub(crate) vae_model: vae::AutoEncoderKL,
    pub(crate) vae_fp16: bool,
    // unet
    pub unet_model: unet_2d::UNet2DConditionModel,
    pub unet_device: Device,
    pub(crate) unet_fp16: bool,
    // 分词器
    pub(crate) tokenizer: Tokenizer,
    // 用于 SDXL
    pub(crate) tokenizer2: Option<Tokenizer>,
    pub(crate) clip_model2: Option<clip::ClipTextTransformer>,
    // 基础模型
    pub base_model: ModelKind,
    // 默认宽高
    pub default_width: usize,
    pub default_height: usize,
    // scheduler
    pub scheduler_kind: SchedulerKind,
}

impl AiyStableDiffusion {
    pub fn new(cfg: AiyConfig) -> anyhow::Result<Self> {
        let clip_device = get_device();
        let unet_device = get_device();
        let vae_device = get_device();
        let bpe = Bpe::new(cfg.vocab_path)?;
        let clip_config = cfg.base_model.clip_config();
        let clip_model = AiyStableDiffusion::build_clip_transformer(
            &clip_config,
            &cfg.clip_weights_path,
            clip_device,
        )?;
        let tokenizer =
            AiyStableDiffusion::create_tokenizer(&bpe, clip_device.clone(), clip_config.clone())?;
        // let clip_config2 = Config::sdxl_v_0_9_encoder2();
        let tokenizer2 = None;
        let clip_model2 = None;
        // VAE
        let vae_model = AiyStableDiffusion::build_vae(
            &cfg.vae_weights_path,
            vae_device,
            cfg.base_model.clone(),
        )?;
        // UNET
        let in_channels = 4;
        let unet_model = AiyStableDiffusion::build_unet(
            &cfg.unet_weights_path,
            unet_device,
            in_channels,
            cfg.base_model.unet_config(),
            &cfg.base_model,
        )?;
        let unet_fp16 = cfg.unet_fp16.unwrap_or(false);
        let vae_fp16 = cfg.vae_fp16.unwrap_or(false);
        let scheduler_kind = cfg.base_model.scheduler_kind();
        Ok(Self {
            tokenizer,
            tokenizer2: tokenizer2,
            clip_model2: clip_model2,
            clip_device,
            unet_device,
            clip_model,
            vae_model,
            vae_device,
            unet_model,
            unet_fp16,
            vae_fp16,
            base_model: cfg.base_model,
            default_height: cfg.height,
            default_width: cfg.width,
            scheduler_kind,
        })
    }

    pub fn create_tokenizer(
        bpe: &Bpe,
        device: Device,
        config: Config,
    ) -> anyhow::Result<Tokenizer> {
        let re = regex::Regex::new(PAT)?;
        let tokenizer = Tokenizer {
            encoder: bpe.encoder.clone(),
            re,
            bpe_ranks: bpe.bpe_ranks.clone(),
            decoder: bpe.decoder.clone(),
            start_of_text_token: bpe.start_of_text_token,
            end_of_text_token: bpe.end_of_text_token,
            config: config,
            device,
        };
        Ok(tokenizer)
    }

    fn build_clip_transformer(
        clip_config: &Config,
        clip_weights: &str,
        device: tch::Device,
    ) -> anyhow::Result<clip::ClipTextTransformer> {
        let mut vs = tch::nn::VarStore::new(device);
        let text_model = clip::ClipTextTransformer::new(
            vs.root(),
            crate::types::SateTensorsFileKind::V0,
            clip_config,
        )?;
        vs.load(clip_weights)?;
        Ok(text_model)
    }

    fn build_vae(
        vae_weights: &str,
        device: Device,
        base_model: ModelKind,
    ) -> anyhow::Result<vae::AutoEncoderKL> {
        let mut vs_ae = tch::nn::VarStore::new(device);
        let autoencoder =
            vae::AutoEncoderKL::new(vs_ae.root(), 3, 3, base_model.vae_config(), base_model);
        vs_ae.load(vae_weights)?;
        vs_ae.set_kind(Kind::Float);
        Ok(autoencoder)
    }

    fn embed_prompts(&self, prompt: &str, negative_prompt: &str) -> anyhow::Result<EmbededPrompts> {
        // Encode prompt and negative prompt
        // 正向提示词
        let tokens = self.tokenizer.parse_prompt(&prompt)?;
        // 负面提示词
        let uncond_tokens = self.tokenizer.parse_prompt(negative_prompt)?;
        let cond_embeddings = self.clip_model.forward(&tokens);
        let uncond_embeddings = self.clip_model.forward(&uncond_tokens);
        let mut text_embeddings;

        // let mut pooled_prompt_embeds = None;
        // let mut negative_pooled_prompt_embeds = None;
        let mut pooled = cond_embeddings.shallow_clone().get(0).get(0);
        let mut uncond_pooled = uncond_embeddings.shallow_clone().get(0).get(0);
        let mut size = cond_embeddings.size();
        if self.base_model.is_sdxl() && self.tokenizer2.is_some() {
            let prompt2 = prompt.clone();
            let negative_prompt2 = negative_prompt.clone();
            let tokens2 = self.tokenizer2.as_ref().unwrap().parse_prompt(&prompt2)?;
            let uncond_tokens2 = self
                .tokenizer2
                .as_ref()
                .unwrap()
                .parse_prompt(negative_prompt2)?;
            let cond_embeddings2 = self.clip_model2.as_ref().unwrap().forward(&tokens2);
            pooled = cond_embeddings2.shallow_clone().get(0).get(0);
            let uncond_embeddings2 = self.clip_model2.as_ref().unwrap().forward(&uncond_tokens2);
            uncond_pooled = uncond_embeddings2.shallow_clone().get(0).get(0);

            // 正向的
            text_embeddings = Tensor::cat(&[cond_embeddings, cond_embeddings2], -1);
            size = text_embeddings.size();

            // 负向的
            let neg_text_embeddings = Tensor::cat(&[uncond_embeddings, uncond_embeddings2], -1);
            // 总的
            text_embeddings = Tensor::cat(&[neg_text_embeddings, text_embeddings], 0);
        } else {
            text_embeddings =
                Tensor::cat(&[uncond_embeddings, cond_embeddings], 0).to(self.unet_device.clone());
        }
        // 处理 pooled_prompt_embeds
        let bs_embed = size.get(0).unwrap().clone();
        let pooled = pooled.repeat(vec![1, 1]);
        let pooled = pooled.view_(vec![bs_embed, -1]);
        let uncond_pooled = uncond_pooled.view_(vec![bs_embed, -1]);
        let pooled_prompt_embeds = Some(pooled);
        let negative_pooled_prompt_embeds = Some(uncond_pooled);
        Ok(EmbededPrompts {
            text_embeddings,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        })
    }

    pub fn vae_decode(&self, t: &Tensor) -> Tensor {
        self.vae_model
            .decode(&(t / self.base_model.scaling_factor()))
    }

    fn build_unet(
        unet_weights: &str,
        device: Device,
        in_channels: i64,
        unet_cfg: unet_2d::UNet2DConditionModelConfig,
        kind: &ModelKind,
    ) -> anyhow::Result<unet_2d::UNet2DConditionModel> {
        let mut vs_unet = tch::nn::VarStore::new(device);
        let unet =
            unet_2d::UNet2DConditionModel::new(vs_unet.root(), in_channels, 4, unet_cfg, kind);
        vs_unet.load(unet_weights)?;
        // vs_unet.set_kind(Kind::Float);
        Ok(unet)
    }

    pub fn text_2_image(
        &self,
        prompt: &str,
        negative_prompt: &str,
        final_image: &str,
        intermediary_images: bool,
        n_steps: usize,
        num_samples: i64,
        seed: i64,
        width: Option<usize>,
        height: Option<usize>,
    ) -> anyhow::Result<()> {
        let EmbededPrompts {
            text_embeddings,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        } = self.embed_prompts(prompt, negative_prompt)?;
        // Scheduler
        let scheduler = self.scheduler_kind.build(n_steps);

        let no_grad_guard = tch::no_grad_guard();
        let bsize = 1;
        let start = SystemTime::now();
        let kind = if self.unet_fp16 {
            Kind::Half
        } else {
            Kind::Float
        };

        let mut add_text_embeds = None;
        if pooled_prompt_embeds.is_some() {
            add_text_embeds = Some(Tensor::concat(
                &[
                    negative_pooled_prompt_embeds.unwrap(),
                    pooled_prompt_embeds.unwrap(),
                ],
                0,
            ));
        }

        println!("---->>>2 {:?}", add_text_embeds);

        for idx in 0..num_samples {
            tch::manual_seed(seed + idx);
            let mut latents = Tensor::randn(
                [
                    bsize,
                    4,
                    (height.unwrap_or(self.default_height) as i64) / 8,
                    (width.unwrap_or(self.default_width) as i64) / 8,
                ],
                (kind, self.unet_device),
            );

            // scale the initial noise by the standard deviation required by the scheduler
            latents *= scheduler.init_noise_sigma();

            let tm = text_embeddings.to_kind(kind);
            let mut start_at;

            for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
                start_at = SystemTime::now();
                let latent_model_input = Tensor::cat(&[&latents, &latents], 0);
                let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep);
                let noise_pred = self.unet_model.forward(
                    &latent_model_input,
                    timestep as f64,
                    &tm,
                    match &add_text_embeds {
                        Some(t) => Some(t.shallow_clone()),
                        None => None,
                    },
                );
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
                    let final_image = output_filename(
                        &final_image,
                        idx + 1,
                        num_samples,
                        Some(timestep_index + 1),
                    );
                    tch::vision::image::save(&image, final_image)?;
                }
                println!(
                    "Step {}/{n_steps}, {:?}",
                    timestep_index + 1,
                    SystemTime::now().duration_since(start_at).unwrap()
                );
            }

            println!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );
            let mut latents = latents.to(self.vae_device);
            // 如果 unet 是 fp16 且 vae 不是 fp16
            if self.unet_fp16 && !self.vae_fp16 {
                latents = latents.to_kind(Kind::Float)
            } else if self.vae_fp16 && !self.unet_fp16 {
                latents = latents.to_kind(Kind::Half)
            }
            latents = latents.to_kind(Kind::Float);

            let image = self.vae_decode(&latents);
            let image = image / 2 + 0.5;
            let image = image.clamp(0., 1.).to_device(Device::Cpu);
            let image = (image * 255.).to_kind(Kind::Uint8);
            let final_image = output_filename(&final_image, idx + 1, num_samples, None);
            tch::vision::image::save(&image, final_image)?;
        }
        println!(
            "=== Generated image: {:?}",
            SystemTime::now().duration_since(start).unwrap()
        );
        drop(no_grad_guard);
        Ok(())
    }

    pub fn image_2_image(
        &self,
        image_path: &str,
        prompt: &str,
        negative_prompt: &str,
        final_image: &str,
        intermediary_images: bool,
        n_steps: usize,
        num_samples: i64,
        seed: i64,
        // The strength, indicates how much to transform the initial image. The
        // value must be between 0 and 1, a value of 1 discards the initial image
        // information.
        // default is 0.8
        strength: Option<f64>,
        // width: Option<usize>,
        // height: Option<usize>,
    ) -> anyhow::Result<()> {
        let strength = strength.unwrap_or(0.8);
        if !(0. ..=1.).contains(&strength) {
            anyhow::bail!("strength should be between 0 and 1, got {strength}")
        }
        let init_image = image_preprocess(image_path)?;
        let EmbededPrompts {
            text_embeddings,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        } = self.embed_prompts(prompt, negative_prompt)?;
        // Scheduler
        let scheduler = if self.base_model.is_sdxl() {
            SchedulerKind::DDIMScheduler(DDIMSchedulerConfig::default()).build(n_steps)
        } else {
            self.scheduler_kind.build(n_steps)
        };

        let no_grad_guard = tch::no_grad_guard();
        // let bsize = 1;
        let start = SystemTime::now();
        let kind = if self.unet_fp16 {
            Kind::Half
        } else {
            Kind::Float
        };

        let mut add_text_embeds = None;
        if pooled_prompt_embeds.is_some() {
            add_text_embeds = Some(Tensor::concat(
                &[
                    negative_pooled_prompt_embeds.unwrap(),
                    pooled_prompt_embeds.unwrap(),
                ],
                0,
            ));
        }

        println!(
            "Generating the latent from the input image {:?}.",
            init_image.size()
        );
        let init_image = init_image.to(self.vae_device);
        println!("---->>> {:?}", init_image);
        let init_latent_dist = self.vae_model.encode(&init_image);

        let t_start = n_steps - (n_steps as f64 * strength) as usize;

        for idx in 0..num_samples {
            tch::manual_seed(seed + idx);
            let latents =
                (init_latent_dist.sample() * self.base_model.scaling_factor()).to(self.unet_device);
            let timesteps = scheduler.timesteps();
            let noise = latents.randn_like();
            let mut latents = scheduler
                .add_noise(&latents, noise, timesteps[t_start])
                .to_kind(kind);

            let tm = text_embeddings.to_kind(kind);
            let mut start_at;

            for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
                start_at = SystemTime::now();
                let latent_model_input = Tensor::cat(&[&latents, &latents], 0);
                let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep);
                let noise_pred = self.unet_model.forward(
                    &latent_model_input,
                    timestep as f64,
                    &tm,
                    match &add_text_embeds {
                        Some(t) => Some(t.shallow_clone()),
                        None => None,
                    },
                );
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
                    let final_image = output_filename(
                        &final_image,
                        idx + 1,
                        num_samples,
                        Some(timestep_index + 1),
                    );
                    tch::vision::image::save(&image, final_image)?;
                }
                println!(
                    "Step {}/{n_steps}, {:?}",
                    timestep_index + 1,
                    SystemTime::now().duration_since(start_at).unwrap()
                );
            }

            println!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );
            let mut latents = latents.to(self.vae_device);
            // 如果 unet 是 fp16 且 vae 不是 fp16
            if self.unet_fp16 && !self.vae_fp16 {
                latents = latents.to_kind(Kind::Float)
            } else if self.vae_fp16 && !self.unet_fp16 {
                latents = latents.to_kind(Kind::Half)
            }
            latents = latents.to_kind(Kind::Float);

            let image = self.vae_decode(&latents);
            let image = image / 2 + 0.5;
            let image = image.clamp(0., 1.).to_device(Device::Cpu);
            let image = (image * 255.).to_kind(Kind::Uint8);
            let final_image = output_filename(&final_image, idx + 1, num_samples, None);
            tch::vision::image::save(&image, final_image)?;
        }
        println!(
            "=== Generated image: {:?}",
            SystemTime::now().duration_since(start).unwrap()
        );
        drop(no_grad_guard);
        Ok(())
    }
}
