use std::path::Path;

use anyhow::Ok;
use tch::{Device, Kind};

use crate::{model_kind::ModelKind, utils::get_device, clip::{bpe::Bpe, Config, ClipTextTransformer}, vae::{self, AutoEncoderKLConfig, AutoEncoderKL}, unet::unet_2d::{self, UNet2DConditionModelConfig, UNet2DConditionModel}, schedulers::SchedulerKind};

use super::{AiyStableDiffusion, config::AiyBaseConfig};

pub struct AiySdBuilder {
    home_dir: String,
    device: Device,
    in_channels: i64,
}

impl AiySdBuilder {
    pub fn new(home_dir: &str) -> Self {
        let device = get_device();
        let in_channels = 4;
        return Self { home_dir: home_dir.to_string(), device, in_channels }
    }

    fn build_clip_model<T: AsRef<std::path::Path>>(&self, model_path: T, cfg: &Config) -> anyhow::Result<ClipTextTransformer> {
        let mut vs = tch::nn::VarStore::new(self.device);
        let text_model = ClipTextTransformer::new(vs.root(), crate::types::SateTensorsFileKind::V0, cfg)?;
        vs.load(model_path)?;
        Ok(text_model)
    }

    fn build_vae_model<T: AsRef<std::path::Path>>(&self, model_path: T, model_kind: &ModelKind, cfg: &AutoEncoderKLConfig) -> anyhow::Result<AutoEncoderKL> {
        let mut vs_ae = tch::nn::VarStore::new(self.device);
        let autoencoder =
            vae::AutoEncoderKL::new(vs_ae.root(), 3, 3, cfg.clone(), model_kind.clone());
        vs_ae.load(model_path)?;
        vs_ae.set_kind(Kind::Float);
        Ok(autoencoder)
    }

    fn build_unet_model<T: AsRef<std::path::Path>>(&self, model_path: T, model_kind: &ModelKind, cfg: &UNet2DConditionModelConfig) -> anyhow::Result<UNet2DConditionModel> {
        let mut vs_unet = tch::nn::VarStore::new(self.device);
        let unet = unet_2d::UNet2DConditionModel::new(vs_unet.root(), self.in_channels, 4, cfg.clone(), model_kind);
        vs_unet.load(model_path)?;
        vs_unet.half();
        Ok(unet)
    }

    pub fn from_repo(&self, repo: &str) -> anyhow::Result<AiyStableDiffusion> {
        let home = Path::new(&self.home_dir);
        let base_root = home.join("__root__");
        let bpe_file = base_root.join("bpe_simple_vocab_16e6.txt");
        // directoies
        let repo = home.join(repo);
        let clip_path = repo.join("clip");
        let clip2_path = repo.join("clip2");

        // aiy_config
        let aiy_config_path = repo.join("config.toml");
        let s = aiy_config_path.clone().to_str().unwrap().to_string();
        let aiy_config = AiyBaseConfig::from_file(aiy_config_path).expect(format!("{:?} not found", s).as_str());

        // toknenizer
        let bpe = Bpe::new(bpe_file.to_str().unwrap().to_string())?;
        let clip_config = Config::from_file(clip_path.join("config.toml"))?;
        let tokenizer = AiyStableDiffusion::create_tokenizer(&bpe, self.device.clone(), clip_config.clone())?;
        let mut tokenizer2 = None;
        
        // clip
        let clip_model_path = clip_path.join("model.fp16.safetensors");
        let clip2_model_path = clip2_path.join("model.fp16.safetensors");
        let clip_model = self.build_clip_model(clip_model_path, &clip_config)?;
        let mut clip2_model = None;

        if aiy_config.model_kind.is_sdxl() {
            let clip_config2 = Config::from_file(clip2_path.join("config.toml")).expect("clip2/config.toml not found");
            tokenizer2 = Some(AiyStableDiffusion::create_tokenizer(&bpe, self.device.clone(), clip_config2.clone())?);
            clip2_model = Some(self.build_clip_model(clip2_model_path, &clip_config2)?);
        }

        // vae
        let vae_path = repo.join("vae");
        let vae_config = AutoEncoderKLConfig::from_file(vae_path.join("config.toml")).expect("Not found config.toml of VAE");
        let vae_model = self.build_vae_model(vae_path.join("diffusion_pytorch_model.fp16.safetensors"), &aiy_config.model_kind, &vae_config)?;

        // unet
        let unet_path = repo.join("unet");
        
        let unet_weights = unet_path.join("diffusion_pytorch_model.fp16.safetensors");
        let unet_config = UNet2DConditionModelConfig::from_file(unet_path.join("config.toml"))?;
        let unet = self.build_unet_model(unet_weights, &aiy_config.model_kind, &unet_config)?;

        // scheduler
        let scheduler_path = repo.join("scheduler");
        let scheduler_kind = SchedulerKind::from_file(scheduler_path.join("config.toml")).expect("scheduler/config.toml not found");

        let vae_fp16 = true;
        let unet_fp16 = true;
        Ok(AiyStableDiffusion {
            scheduler_kind,
            clip_device: self.device,
            clip_model: clip_model,
            vae_device: self.device,
            vae_model,
            vae_fp16,
            unet_model: unet,
            unet_device: self.device,
            unet_fp16,
            tokenizer,
            tokenizer2,
            clip_model2: clip2_model,
            base_model: aiy_config.model_kind,
            default_width: aiy_config.default_width,
            default_height: aiy_config.default_height,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::AiySdBuilder;

    #[test]
    fn test_sdxl_0_9() {
        let builder = AiySdBuilder::new("E://.aiy-repos");
        let aiy = builder.from_repo("sdxl-base-1.0").unwrap();
        // let bpe_file = "data/bpe_simple_vocab_16e6.txt";
        // let file_path = "D:\\stable diffusion\\stable-diffusion-webui-1.6.0\\models\\Stable-diffusion\\sd_xl_base_0.9.f16.safetensors";
        // let base_model = crate::model_kind::ModelKind::SDXL_0_9;
        // let aiy = AiySdBuilder::from_single_weights_file(bpe_file, file_path, base_model).unwrap();
        aiy.text_2_image(
            // "A very realistic photo of a rusty robot walking on a sandy beach, sea and a green tree",
            // "A high-resolution photograph of a waterfall in autumn; muted tone",
            // "An astronaut riding a green horse",
            // "",
            "ultra realistic close up portrait ((beautiful pale cyberpunk female with heavy black eyeliner)), blue eyes, shaved side haircut, hyper detail, cinematic lighting, magic neon, dark red city, Canon EOS R3, her is touch her hair, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame, 8K",
            "painting, extra fingers, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
            "./sd_final.png",
            false,
            20,
            1,
            32,
            None,
            None
        ).unwrap();
    }
}
