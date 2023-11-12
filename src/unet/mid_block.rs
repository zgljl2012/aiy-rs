use tch::{nn, Tensor};

use crate::{resnet::{ResnetBlock2D, ResnetBlock2DConfig}, attention::{AttentionBlock, SpatialTransformer, SpatialTransformerConfig, AttentionBlockConfig}, model_kind::ModelKind};

#[derive(Debug, Clone, Copy)]
pub struct UNetMidBlock2DConfig {
    pub num_layers: i64,
    pub resnet_eps: f64,
    pub resnet_groups: Option<i64>,
    pub attn_num_head_channels: Option<i64>,
    // attention_type "default"
    pub output_scale_factor: f64,
}

impl Default for UNetMidBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: Some(32),
            attn_num_head_channels: Some(1),
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
pub struct UNetMidBlock2D {
    resnet: ResnetBlock2D,
    attn_resnets: Vec<(AttentionBlock, ResnetBlock2D)>,
    pub config: UNetMidBlock2DConfig,
}

impl UNetMidBlock2D {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        temb_channels: Option<i64>,
        config: UNetMidBlock2DConfig,
        base_model: ModelKind
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let vs_attns = &vs / "attentions";
        let resnet_groups = config.resnet_groups.unwrap_or_else(|| i64::min(in_channels / 4, 32));
        let resnet_cfg = ResnetBlock2DConfig {
            eps: config.resnet_eps,
            groups: resnet_groups,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnet = ResnetBlock2D::new(&vs_resnets / "0", in_channels, resnet_cfg);
        let attn_cfg = AttentionBlockConfig {
            num_head_channels: config.attn_num_head_channels,
            num_groups: resnet_groups,
            rescale_output_factor: config.output_scale_factor,
            eps: config.resnet_eps,
        };
        let mut attn_resnets = vec![];
        for index in 0..config.num_layers {
            let attn = AttentionBlock::new(&vs_attns / index, in_channels, attn_cfg, base_model.clone());
            let resnet = ResnetBlock2D::new(&vs_resnets / (index + 1), in_channels, resnet_cfg);
            attn_resnets.push((attn, resnet))
        }
        Self { resnet, attn_resnets, config }
    }

    pub fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Tensor {
        let mut xs = self.resnet.forward(xs, temb);
        for (attn, resnet) in self.attn_resnets.iter() {
            xs = resnet.forward(&xs.apply(attn), temb)
        }
        xs
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UNetMidBlock2DCrossAttnConfig {
    pub num_layers: i64,
    pub resnet_eps: f64,
    pub resnet_groups: Option<i64>,
    pub attn_num_head_channels: i64,
    // attention_type "default"
    pub output_scale_factor: f64,
    pub cross_attn_dim: i64,
    pub sliced_attention_size: Option<i64>,
    pub use_linear_projection: bool,
}

impl Default for UNetMidBlock2DCrossAttnConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: Some(32),
            attn_num_head_channels: 1,
            output_scale_factor: 1.,
            cross_attn_dim: 1280,
            sliced_attention_size: None, // Sliced attention disabled
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
pub struct UNetMidBlock2DCrossAttn {
    resnet: ResnetBlock2D,
    attn_resnets: Vec<(SpatialTransformer, ResnetBlock2D)>,
    pub config: UNetMidBlock2DCrossAttnConfig,
}

impl UNetMidBlock2DCrossAttn {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        temb_channels: Option<i64>,
        config: UNetMidBlock2DCrossAttnConfig,
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let vs_attns = &vs / "attentions";
        let resnet_groups = config.resnet_groups.unwrap_or_else(|| i64::min(in_channels / 4, 32));
        let resnet_cfg = ResnetBlock2DConfig {
            eps: config.resnet_eps,
            groups: resnet_groups,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnet = ResnetBlock2D::new(&vs_resnets / "0", in_channels, resnet_cfg);
        let n_heads = config.attn_num_head_channels;
        let attn_cfg = SpatialTransformerConfig {
            depth: 1,
            num_groups: resnet_groups,
            context_dim: Some(config.cross_attn_dim),
            sliced_attention_size: config.sliced_attention_size,
            use_linear_projection: config.use_linear_projection,
        };
        let mut attn_resnets = vec![];
        for index in 0..config.num_layers {
            let attn = SpatialTransformer::new(
                &vs_attns / index,
                in_channels,
                n_heads,
                in_channels / n_heads,
                attn_cfg,
            );
            let resnet = ResnetBlock2D::new(&vs_resnets / (index + 1), in_channels, resnet_cfg);
            attn_resnets.push((attn, resnet))
        }
        Self { resnet, attn_resnets, config }
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Tensor {
        let mut xs = self.resnet.forward(xs, temb);
        for (attn, resnet) in self.attn_resnets.iter() {
            xs = resnet.forward(&attn.forward(&xs, encoder_hidden_states), temb)
        }
        xs
    }
}
