use std::time::SystemTime;

use aiy_rs::{
    aiy_sd::{AiyConfig, AiyStableDiffusion},
    clip_configs::ClipConfig,
    unet_config::UNetConfig, model_kind::ModelKind,
};

// use diffusers::schedulers::PredictionType;

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
    }
}

fn run() -> anyhow::Result<()> {
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    println!("MPS available: {}", tch::utils::has_mps());

    let start = SystemTime::now();

    let aiy = AiyStableDiffusion::new(sdv1_5())
    .unwrap();

    println!(
        "=== Device setup: {:?}",
        SystemTime::now().duration_since(start).unwrap()
    );

    aiy.run(
        "A very realistic photo of a rusty robot walking on a sandy beach",
        "sea, tiny eye, half body",
        "./sd_final.png",
        false,
        30,
        1,
        32,
        512,
        512,
        None
        // Some(PredictionType::VPrediction),
    )?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    run()
}
