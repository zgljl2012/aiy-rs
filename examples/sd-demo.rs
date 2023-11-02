use std::time::SystemTime;

use aiy_rs::{
    aiy_sd::{AiyConfig, AiyStableDiffusion},
    clip_configs::ClipConfig,
    unet_config::UNetConfig, model_kind::ModelKind,
};

use diffusers::schedulers::PredictionType;

#[allow(dead_code)]
fn sdv1_5() -> AiyConfig {
    AiyConfig {
        vocab_path: "data/bpe_simple_vocab_16e6.txt".to_string(),
        // CLIP
        clip_weights_path: "data/clip_v1.5.fp16.safetensors".to_string(),
        clip_config: ClipConfig::V1_5.config(),
        // VAE
        vae_weights_path: "data/vae_v1.5.fp16.safetensors".to_string(),
        vae_fp16: Some(true),
        // UNET
        unet_weights_path: "data/unet_v1.5.fp16.safetensors".to_string(),
        unet_fp16: Some(true),
        unet_config: UNetConfig::V1_5.config(),
        // 指定基础模型
        base_model: ModelKind::SD1_5,
        // width & height
        width: 512,
        height: 512,
        prediction_type: None
    }
}

/// https://huggingface.co/jzli/BRA-v6
#[allow(dead_code)]
fn bra_v6() -> AiyConfig {
    AiyConfig {
        vocab_path: "data/bpe_simple_vocab_16e6.txt".to_string(),
        // CLIP
        clip_weights_path: "data/brav6-clip.safetensors".to_string(),
        clip_config: ClipConfig::V1_5.config(),
        // VAE
        vae_weights_path: "data/brav6-vae.safetensors".to_string(),
        vae_fp16: Some(false),
        // UNET
        unet_weights_path: "data/brav6-unet.safetensors".to_string(),
        unet_fp16: Some(false),
        unet_config: UNetConfig::V1_5.config(),
        // 指定基础模型
        base_model: ModelKind::SD2_1, // VAE 取值 Path
        // width & height
        width: 512,
        height: 512,
        prediction_type: None
    }
}

/// https://huggingface.co/stablediffusionapi/beautiful-realistic-asian
#[allow(dead_code)]
fn bra_v7() -> AiyConfig {
    AiyConfig {
        vocab_path: "data/bpe_simple_vocab_16e6.txt".to_string(),
        // CLIP
        clip_weights_path: "data/brav7-clip.safetensors".to_string(),
        clip_config: ClipConfig::V1_5.config(),
        // VAE
        vae_weights_path: "data/brav7-vae.safetensors".to_string(),
        vae_fp16: Some(true),
        // UNET
        unet_weights_path: "data/brav7-unet.safetensors".to_string(),
        unet_fp16: Some(true),
        unet_config: UNetConfig::V1_5.config(),
        // 指定基础模型
        base_model: ModelKind::SD1_5, // VAE 取值 Path
        // width & height
        width: 512,
        height: 512,
        prediction_type: None
    }
}

#[allow(dead_code)]
fn sdv2_1() -> AiyConfig {
    AiyConfig {
        vocab_path: "data/bpe_simple_vocab_16e6.txt".to_string(),
        // CLIP
        clip_weights_path: "data/clip_v2.1.fp16.safetensors".to_string(),
        clip_config: ClipConfig::V2_1.config(),
        // VAE
        vae_weights_path: "data/vae_v2.1.fp16.safetensors".to_string(),
        vae_fp16: Some(true),
        // UNET
        unet_weights_path: "data/unet_v2.1.fp16.safetensors".to_string(),
        unet_fp16: Some(true),
        unet_config: UNetConfig::V2_1.config(),
        // 指定基础模型
        base_model: ModelKind::SD2_1,
        // width & height
        width: 768,
        height: 768,
        prediction_type: Some(PredictionType::VPrediction)
    }
}


fn run() -> anyhow::Result<()> {
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    println!("MPS available: {}", tch::utils::has_mps());

    let start = SystemTime::now();

    let aiy = AiyStableDiffusion::new(bra_v7())
    .unwrap();

    println!(
        "=== Device setup: {:?}",
        SystemTime::now().duration_since(start).unwrap()
    );

    aiy.text_2_image(
        // "A very realistic photo of a rusty robot walking on a sandy beach",
        "ultra realistic close up portrait ((beautiful pale cyberpunk female with heavy black eyeliner)), blue eyes, shaved side haircut, hyper detail, cinematic lighting, magic neon, dark red city, Canon EOS R3, her is touch her hair, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame, 8K",
        "painting, extra fingers, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
        "./sd_final.png",
        false,
        30,
        1,
        32,
        None,
        None
    )?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    run()
}
