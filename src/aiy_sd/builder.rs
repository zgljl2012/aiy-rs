use std::path::Path;

use anyhow::Ok;
use tch::Device;

use crate::{model_kind::ModelKind, utils::get_device, clip::{self, bpe::Bpe, Config, ClipTextTransformer}, vae, unet::unet_2d, types::SateTensorsFileKind};

use super::AiyStableDiffusion;

pub struct AiySdBuilder {
    home_dir: String,
    device: Device
}

impl AiySdBuilder {
    pub fn new(home_dir: &str) -> Self {
        let device = get_device();
        return Self { home_dir: home_dir.to_string(), device }
    }

    fn build_clip_clip_model<T: AsRef<std::path::Path>>(&self, model_path: T, cfg: &Config) -> anyhow::Result<ClipTextTransformer> {
        let mut vs = tch::nn::VarStore::new(self.device);
        let text_model = ClipTextTransformer::new(vs.root(), crate::types::SateTensorsFileKind::V0, cfg)?;
        vs.load(model_path)?;
        Ok(text_model)
    }

    pub fn from_repo(&self, repo: &str) -> anyhow::Result<()> {
        let home = Path::new(&self.home_dir);
        let base_root = home.join("__root__");
        let bpe_file = base_root.join("bpe_simple_vocab_16e6.txt");
        // directoies
        let repo = home.join(repo);
        let clip_path = repo.join("clip");
        let clip2_path = repo.join("clip2");

        // toknenizer
        let bpe = Bpe::new(bpe_file.to_str().unwrap().to_string())?;
        let clip_config = Config::from_file(clip_path.join("config.toml"))?;
        let clip_config2 = Config::from_file(clip2_path.join("config.toml"))?;
        let tokenizer = AiyStableDiffusion::create_tokenizer(&bpe, self.device.clone(), clip_config.clone())?;
        let tokenizer2 = AiyStableDiffusion::create_tokenizer(&bpe, self.device.clone(), clip_config2.clone())?;
        
        // clip
        let clip_model_path = clip_path.join("model.fp16.safetensors");
        let clip2_model_path = clip2_path.join("model.fp16.safetensors");
        let clip_model = self.build_clip_clip_model(clip_model_path, &clip_config)?;
        let clip2_model = self.build_clip_clip_model(clip2_model_path, &clip_config2)?;



        Ok(())
    }
    pub fn from_single_weights_file(bpe_file: &str, file_path: &str, base_model: ModelKind) -> anyhow::Result<AiyStableDiffusion> {
        let device = get_device();
        let mut vs = tch::nn::VarStore::new(device);
        let vae_fp16 = true;
        let unet_fp16 = true;
        
        // tokenizer
        let bpe = Bpe::new(bpe_file.to_string())?;
        let clip_config = base_model.clip_config();
        let clip_config2 = base_model.clip_config2();
        let tokenizer = AiyStableDiffusion::create_tokenizer(&bpe, device.clone(), clip_config.clone())?;
        let tokenizer2 = AiyStableDiffusion::create_tokenizer(&bpe, device.clone(), clip_config2.clone())?;

        // clip 1
        let clip_vs_root = vs.root() / "conditioner" / "embedders" / 0 / "transformer";
        let clip_model_1 = clip::ClipTextTransformer::new(clip_vs_root, SateTensorsFileKind::V0, &clip_config)?;
        
        // clip 2
        let clip_vs_root = vs.root() / "conditioner" / "embedders" / 1 / "model";
        let clip_model_2 = clip::ClipTextTransformer::new(clip_vs_root, SateTensorsFileKind::V1,&clip_config2)?;

        // vae
        let vae_model =
            vae::AutoEncoderKL::new(vs.root() / "first-stage-model", 3, 3, base_model.vae_config(), base_model.clone());
        
        // unet
        let in_channels = 4;
        vs.half();
        let unet = unet_2d::UNet2DConditionModel::new(vs.root() / "model" / "diffusion_model", in_channels, 4, base_model.unet_config(), &base_model);

        vs.load(file_path)?;
        Ok(AiyStableDiffusion {
            clip_device: device,
            clip_model: clip_model_1,
            vae_device: device,
            vae_model,
            vae_fp16,
            unet_model: unet,
            unet_device: device,
            unet_fp16,
            tokenizer,
            tokenizer2,
            clip_model2: clip_model_2,
            base_model,
            default_width: 1024,
            default_height: 1024,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::AiySdBuilder;

    #[test]
    fn test_sdxl_0_9() {
        let builder = AiySdBuilder::new("E://.aiy-repos");
        builder.from_repo("sdxl-base-0.9").unwrap();
        // let bpe_file = "data/bpe_simple_vocab_16e6.txt";
        // let file_path = "D:\\stable diffusion\\stable-diffusion-webui-1.6.0\\models\\Stable-diffusion\\sd_xl_base_0.9.f16.safetensors";
        // let base_model = crate::model_kind::ModelKind::SDXL_0_9;
        // let aiy = AiySdBuilder::from_single_weights_file(bpe_file, file_path, base_model).unwrap();
        // aiy.text_2_image(
        //     // "A very realistic photo of a rusty robot walking on a sandy beach, sea and a green tree",
        //     // "A high-resolution photograph of a waterfall in autumn; muted tone",
        //     // "An astronaut riding a green horse",
        //     // "",
        //     "ultra realistic close up portrait ((beautiful pale cyberpunk female with heavy black eyeliner)), blue eyes, shaved side haircut, hyper detail, cinematic lighting, magic neon, dark red city, Canon EOS R3, her is touch her hair, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame, 8K",
        //     "painting, extra fingers, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
        //     "./sd_final.png",
        //     false,
        //     20,
        //     1,
        //     32,
        //     None,
        //     None
        // ).unwrap();
    }
}
