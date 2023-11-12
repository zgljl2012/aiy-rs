use tch::nn;

use crate::{
    clip::Config,
    schedulers::{ddim::DDIMSchedulerConfig, PredictionType, SchedulerKind, euler_discrete::EulerDiscreteSchedulerConfig},
    unet::{unet_2d, unet_config::UNetConfig},
    vae,
};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub enum ModelKind {
    SD1_5,
    SD2_1,
    SDXL_0_9,
}

pub struct VaeAttensionPaths<'a> {
    pub group_norm: nn::Path<'a>,
    pub query: nn::Path<'a>,
    pub key: nn::Path<'a>,
    pub value: nn::Path<'a>,
    pub proj_attn: nn::Path<'a>,
}

impl ModelKind {
    pub fn is_sdxl(&self) -> bool {
        match &self {
            ModelKind::SDXL_0_9 => true,
            _ => false,
        }
    }

    pub fn clip_config(&self) -> Config {
        match self {
            ModelKind::SD1_5 => Config::v1_5(),
            ModelKind::SD2_1 => Config::v2_1(),
            ModelKind::SDXL_0_9 => Config::sdxl_v_0_9(),
        }
    }

    pub fn vae_config(&self) -> vae::AutoEncoderKLConfig {
        let autoencoder = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
        };
        return autoencoder;
    }

    pub fn unet_config(&self) -> unet_2d::UNet2DConditionModelConfig {
        match &self {
            ModelKind::SD1_5 => UNetConfig::V1_5.config(),
            ModelKind::SD2_1 => UNetConfig::V2_1.config(),
            ModelKind::SDXL_0_9 => {
                let bc =
                    |out_channels,
                     use_cross_attn,
                     attention_head_dim,
                     transformer_layers_per_block| unet_2d::BlockConfig {
                        out_channels,
                        use_cross_attn,
                        attention_head_dim,
                        transformer_layers_per_block,
                    };
                // The size of the sliced attention or 0 for automatic slicing (disabled by default)
                let sliced_attention_size = None;
                let unet = unet_2d::UNet2DConditionModelConfig {
                    blocks: vec![
                        bc(320, false, 5, 1),
                        bc(640, true, 10, 2),
                        bc(1280, true, 20, 10),
                    ],
                    center_input_sample: false,
                    cross_attention_dim: 2048,
                    downsample_padding: 1,
                    flip_sin_to_cos: true,
                    freq_shift: 0.,
                    layers_per_block: 2,
                    mid_block_scale_factor: 1.,
                    norm_eps: 1e-5,
                    norm_num_groups: 32,
                    sliced_attention_size,
                    use_linear_projection: true,
                };
                return unet;
            }
        }
    }

    pub fn vae_attension_paths<'a>(&self, v: nn::Path<'a>) -> VaeAttensionPaths<'a> {
        let v1_5: VaeAttensionPaths<'a> = VaeAttensionPaths {
            group_norm: &v / "group_norm",
            query: &v / "to_q",
            key: &v / "to_k",
            value: &v / "to_v",
            proj_attn: &v / "to_out" / 0,
        };
        let v2_1: VaeAttensionPaths<'a> = VaeAttensionPaths {
            group_norm: &v / "group_norm",
            query: &v / "query",
            key: &v / "key",
            value: &v / "value",
            proj_attn: &v / "proj_attn",
        };
        match &self {
            ModelKind::SD1_5 => v1_5,
            ModelKind::SD2_1 => v2_1,
            ModelKind::SDXL_0_9 => v1_5,
        }
    }

    pub fn scheduler_kind(&self) -> SchedulerKind {
        match &self {
            ModelKind::SDXL_0_9 => SchedulerKind::EulerDiscreteScheduler(EulerDiscreteSchedulerConfig {
                
                ..Default::default()
            }),
            ModelKind::SD2_1 => SchedulerKind::DDIMScheduler(DDIMSchedulerConfig {
                prediction_type: PredictionType::VPrediction,
                ..Default::default()
            }),
            _ => SchedulerKind::DDIMScheduler(DDIMSchedulerConfig {
                ..Default::default()
            }),
        }
    }
}
