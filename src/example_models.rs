
use crate::{
    aiy_sd::AiyConfig,
    clip_configs::ClipConfig,
    unet_config::UNetConfig, model_kind::ModelKind,
};

use diffusers::schedulers::PredictionType;


#[allow(dead_code)]
pub fn sdv1_5() -> AiyConfig {
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
pub fn bra_v6() -> AiyConfig {
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
pub fn bra_v7() -> AiyConfig {
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
pub fn sdv2_1() -> AiyConfig {
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