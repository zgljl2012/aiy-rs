use std::time::SystemTime;

use aiy_rs::{
    aiy_sd::{AiyConfig, AiyStableDiffusion},
    clip_configs::ClipConfig,
    unet_config::UNetConfig,
};

use diffusers::schedulers::PredictionType;

fn run() -> anyhow::Result<()> {
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    println!("MPS available: {}", tch::utils::has_mps());

    let start = SystemTime::now();

    let aiy = AiyStableDiffusion::new(AiyConfig {
        vocab_path: "data/bpe_simple_vocab_16e6.txt".to_string(),
        // CLIP
        clip_weights_path: "data/clip_v2.1.safetensors".to_string(),
        clip_config: ClipConfig::V2_1.config(),
        // VAE
        vae_weights_path: "data/vae_v2.1.safetensors".to_string(),
        // UNET
        unet_weights_path: "data/unet_v2.1.safetensors".to_string(),
        unet_config: UNetConfig::V2_1.config(),
    })
    .unwrap();

    println!(
        "=== Device setup: {:?}",
        SystemTime::now().duration_since(start).unwrap()
    );

    aiy.run(
        "A very realistic photo of a rusty robot walking on a sandy beach",
        "sea, tiny eye",
        "./sd_final.png",
        false,
        30,
        1,
        32,
        768,
        768,
        Some(PredictionType::VPrediction),
    )?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    run()
}
