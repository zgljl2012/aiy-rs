use tch::{nn::{self, Module}, Tensor};

use crate::{unet::{unet_2d_blocks::{UpDecoderBlock2D, UpDecoderBlock2DConfig}, UNetMidBlock2D, UNetMidBlock2DConfig}, model_kind::ModelKind};

#[derive(Debug, Clone)]
pub struct DecoderConfig {
    // up_block_types: UpDecoderBlock2D
    pub block_out_channels: Vec<i64>,
    pub layers_per_block: i64,
    pub norm_num_groups: i64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self { block_out_channels: vec![64], layers_per_block: 2, norm_num_groups: 32 }
    }
}

#[derive(Debug)]
pub struct Decoder {
    conv_in: nn::Conv2D,
    up_blocks: Vec<UpDecoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    #[allow(dead_code)]
    config: DecoderConfig,
}

impl Decoder {
    pub fn new(vs: nn::Path, in_channels: i64, out_channels: i64, config: DecoderConfig, base_model: ModelKind) -> Self {
        let n_block_out_channels = config.block_out_channels.len();
        let last_block_out_channels = *config.block_out_channels.last().unwrap();
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in =
            nn::conv2d(&vs / "conv_in", in_channels, last_block_out_channels, 3, conv_cfg);
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block =
            UNetMidBlock2D::new(&vs / "mid_block", last_block_out_channels, None, mid_cfg, base_model);
        let mut up_blocks = vec![];
        let vs_up_blocks = &vs / "up_blocks";
        let reversed_block_out_channels: Vec<_> =
            config.block_out_channels.iter().copied().rev().collect();
        for index in 0..n_block_out_channels {
            let out_channels = reversed_block_out_channels[index];
            let in_channels = if index > 0 {
                reversed_block_out_channels[index - 1]
            } else {
                reversed_block_out_channels[0]
            };
            let is_final = index + 1 == n_block_out_channels;
            let cfg = UpDecoderBlock2DConfig {
                num_layers: config.layers_per_block + 1,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_upsample: !is_final,
                output_scale_factor: 1.,
                ..Default::default()
            };
            let up_block =
                UpDecoderBlock2D::new(&vs_up_blocks / index, in_channels, out_channels, cfg);
            up_blocks.push(up_block)
        }
        let group_cfg = nn::GroupNormConfig { eps: 1e-6, ..Default::default() };
        let conv_norm_out = nn::group_norm(
            &vs / "conv_norm_out",
            config.norm_num_groups,
            config.block_out_channels[0],
            group_cfg,
        );
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv_out =
            nn::conv2d(&vs / "conv_out", config.block_out_channels[0], out_channels, 3, conv_cfg);
        Self { conv_in, up_blocks, mid_block, conv_norm_out, conv_out, config }
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // 先卷积
        let t = xs.apply(&self.conv_in);
        // 经历 mid block
        let mut xs = self.mid_block.forward(&t, None);
        // 依次向上 block
        for up_block in self.up_blocks.iter() {
            xs = xs.apply(up_block);
        }
        // 卷积且输出图像
        xs.apply(&self.conv_norm_out).silu().apply(&self.conv_out)
    }
}
