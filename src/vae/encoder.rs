use crate::{
    model_kind::ModelKind,
    unet::unet_2d_blocks::{
        DownEncoderBlock2D, DownEncoderBlock2DConfig, UNetMidBlock2D, UNetMidBlock2DConfig,
    },
};
use tch::{nn, nn::Module, Tensor};

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    // down_block_types: DownEncoderBlock2D
    pub block_out_channels: Vec<i64>,
    pub layers_per_block: i64,
    pub norm_num_groups: i64,
    pub double_z: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 2,
            norm_num_groups: 32,
            double_z: true,
        }
    }
}

#[derive(Debug)]
pub struct Encoder {
    conv_in: nn::Conv2D,
    down_blocks: Vec<DownEncoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    #[allow(dead_code)]
    config: EncoderConfig,
}

impl Encoder {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: EncoderConfig,
        base_model: ModelKind,
    ) -> Self {
        let conv_cfg = nn::ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let conv_in = nn::conv2d(
            &vs / "conv_in",
            in_channels,
            config.block_out_channels[0],
            3,
            conv_cfg,
        );
        let mut down_blocks = vec![];
        let vs_down_blocks = &vs / "down_blocks";
        for index in 0..config.block_out_channels.len() {
            let out_channels = config.block_out_channels[index];
            let in_channels = if index > 0 {
                config.block_out_channels[index - 1]
            } else {
                config.block_out_channels[0]
            };
            let is_final = index + 1 == config.block_out_channels.len();
            let cfg = DownEncoderBlock2DConfig {
                num_layers: config.layers_per_block,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_downsample: !is_final,
                downsample_padding: 0,
                ..Default::default()
            };
            let down_block =
                DownEncoderBlock2D::new(&vs_down_blocks / index, in_channels, out_channels, cfg);
            down_blocks.push(down_block)
        }
        let last_block_out_channels = *config.block_out_channels.last().unwrap();
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block = UNetMidBlock2D::new(
            &vs / "mid_block",
            last_block_out_channels,
            None,
            mid_cfg,
            base_model,
        );
        let group_cfg = nn::GroupNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let conv_norm_out = nn::group_norm(
            &vs / "conv_norm_out",
            config.norm_num_groups,
            last_block_out_channels,
            group_cfg,
        );
        let conv_out_channels = if config.double_z {
            2 * out_channels
        } else {
            out_channels
        };
        let conv_cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_out = nn::conv2d(
            &vs / "conv_out",
            last_block_out_channels,
            conv_out_channels,
            3,
            conv_cfg,
        );
        Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
            config,
        }
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.apply(&self.conv_in);
        for down_block in self.down_blocks.iter() {
            xs = xs.apply(down_block)
        }
        self.mid_block
            .forward(&xs, None)
            .apply(&self.conv_norm_out)
            .silu()
            .apply(&self.conv_out)
    }
}
