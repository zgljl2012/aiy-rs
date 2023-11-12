//! 2D UNet Denoising Models
//!
//! The 2D Unet models take as input a noisy sample and the current diffusion
//! timestep and return a denoised version of the input.
use crate::{embeddings::{TimestepEmbedding, Timesteps}, model_kind::ModelKind};
use super::{unet_2d_blocks::*, UNetMidBlock2DCrossAttn, UNetMidBlock2DCrossAttnConfig};
use tch::{nn::{self, Module}, Tensor, Kind};

#[derive(Debug, Clone, Copy)]
pub struct BlockConfig {
    pub out_channels: i64,
    pub use_cross_attn: bool,
    pub attention_head_dim: i64,
    pub transformer_layers_per_block: i64,
}

#[derive(Debug, Clone)]
pub struct UNet2DConditionModelConfig {
    pub center_input_sample: bool,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f64,
    pub blocks: Vec<BlockConfig>,
    pub layers_per_block: i64,
    pub downsample_padding: i64,
    pub mid_block_scale_factor: f64,
    pub norm_num_groups: i64,
    pub norm_eps: f64,
    pub cross_attention_dim: i64,
    pub sliced_attention_size: Option<i64>,
    pub use_linear_projection: bool,
}

impl Default for UNet2DConditionModelConfig {
    fn default() -> Self {
        Self {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![
                BlockConfig { out_channels: 320, use_cross_attn: true, attention_head_dim: 8, transformer_layers_per_block: 1 },
                BlockConfig { out_channels: 640, use_cross_attn: true, attention_head_dim: 8, transformer_layers_per_block: 1 },
                BlockConfig { out_channels: 1280, use_cross_attn: true, attention_head_dim: 8, transformer_layers_per_block: 1 },
                BlockConfig { out_channels: 1280, use_cross_attn: false, attention_head_dim: 8, transformer_layers_per_block: 1 },
            ],
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 1280,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
pub(crate) enum UNetDownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

#[derive(Debug)]
enum UNetUpBlock {
    Basic(UpBlock2D),
    CrossAttn(CrossAttnUpBlock2D),
}

#[derive(Debug)]
pub struct UNet2DConditionModel {
    conv_in: nn::Conv2D,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    // SDXL 有此 embedding
    add_time_proj: Option<Timesteps>,
    add_embedding: Option<TimestepEmbedding>,
    down_blocks: Vec<UNetDownBlock>,
    mid_block: UNetMidBlock2DCrossAttn,
    up_blocks: Vec<UNetUpBlock>,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    config: UNet2DConditionModelConfig,
}

impl UNet2DConditionModel {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: UNet2DConditionModelConfig,
        kind: &ModelKind
    ) -> Self {
        let n_blocks = config.blocks.len();
        let b_channels = config.blocks[0].out_channels;
        let bl_channels = config.blocks.last().unwrap().out_channels;
        let bl_attention_head_dim = config.blocks.last().unwrap().attention_head_dim;
        let time_embed_dim = b_channels * 4;
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in = nn::conv2d(&vs / "conv_in", in_channels, b_channels, 3, conv_cfg);

        let time_proj =
            Timesteps::new(b_channels, config.flip_sin_to_cos, config.freq_shift, vs.device());
        let time_embedding =
            TimestepEmbedding::new(&vs / "time_embedding", b_channels, time_embed_dim);
        // TODO 处理 XL 和 1.5 2.1 的区别
        let addition_time_embed_dim = 256;
        let mut add_time_proj = None;
        let mut add_embedding = None;
        if kind.is_sdxl() {
            add_time_proj = Some(Timesteps::new(addition_time_embed_dim, config.flip_sin_to_cos, config.freq_shift, vs.device()));
            add_embedding = Some(TimestepEmbedding::new(&vs / "add_embedding", 2816, time_embed_dim));
        }
        
        
        let vs_db = &vs / "down_blocks";
        // 读取 down_blocks
        let down_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig { out_channels, use_cross_attn, attention_head_dim, transformer_layers_per_block } =
                    config.blocks[i];

                // Enable automatic attention slicing if the config sliced_attention_size is set to 0.
                let sliced_attention_size = match config.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    _ => config.sliced_attention_size,
                };

                let in_channels =
                    if i > 0 { config.blocks[i - 1].out_channels } else { b_channels };
                let db_cfg = DownBlock2DConfig {
                    num_layers: config.layers_per_block,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_downsample: i < n_blocks - 1,
                    downsample_padding: config.downsample_padding,
                    ..Default::default()
                };
                if use_cross_attn {
                    let config = CrossAttnDownBlock2DConfig {
                        depth: transformer_layers_per_block,
                        downblock: db_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                    };
                    let block = CrossAttnDownBlock2D::new(
                        &vs_db / i,
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        config,
                    );
                    UNetDownBlock::CrossAttn(block)
                } else {
                    let block = DownBlock2D::new(
                        &vs_db / i,
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        db_cfg,
                    );
                    UNetDownBlock::Basic(block)
                }
            })
            .collect();

        let mid_depth = config.blocks.last().unwrap().transformer_layers_per_block;
        let mid_cfg = UNetMidBlock2DCrossAttnConfig {
            depth: mid_depth,
            resnet_eps: config.norm_eps,
            output_scale_factor: config.mid_block_scale_factor,
            cross_attn_dim: config.cross_attention_dim,
            attn_num_head_channels: bl_attention_head_dim,
            resnet_groups: Some(config.norm_num_groups),
            use_linear_projection: config.use_linear_projection,
            ..Default::default()
        };
        let mid_block = UNetMidBlock2DCrossAttn::new(
            &vs / "mid_block",
            bl_channels,
            Some(time_embed_dim),
            mid_cfg,
        );

        let vs_ub = &vs / "up_blocks";
        // 读取 up_blocks 
        let up_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig { out_channels, use_cross_attn, attention_head_dim, transformer_layers_per_block } =
                    config.blocks[n_blocks - 1 - i]; // blocks 倒过来就是 up_blocks，正着数就是 down_blocks

                // Enable automatic attention slicing if the config sliced_attention_size is set to 0.
                let sliced_attention_size = match config.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    _ => config.sliced_attention_size,
                };

                let prev_out_channels =
                    if i > 0 { config.blocks[n_blocks - i].out_channels } else { bl_channels };
                let in_channels = {
                    let index = if i == n_blocks - 1 { 0 } else { n_blocks - i - 2 };
                    config.blocks[index].out_channels
                };
                let ub_cfg = UpBlock2DConfig {
                    num_layers: config.layers_per_block + 1,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_upsample: i < n_blocks - 1,
                    ..Default::default()
                };
                if use_cross_attn {
                    let config = CrossAttnUpBlock2DConfig {
                        depth: transformer_layers_per_block,
                        upblock: ub_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                    };
                    let block = CrossAttnUpBlock2D::new(
                        &vs_ub / i,
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        config,
                    );
                    UNetUpBlock::CrossAttn(block)
                } else {
                    let block = UpBlock2D::new(
                        &vs_ub / i,
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        ub_cfg,
                    );
                    UNetUpBlock::Basic(block)
                }
            })
            .collect();

        let group_cfg = nn::GroupNormConfig { eps: config.norm_eps, ..Default::default() };
        let conv_norm_out =
            nn::group_norm(&vs / "conv_norm_out", config.norm_num_groups, b_channels, group_cfg);
        let conv_out = nn::conv2d(&vs / "conv_out", b_channels, out_channels, 3, conv_cfg);
        Self {
            conv_in,
            time_proj,
            time_embedding,
            add_time_proj,
            add_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            config,
        }
    }
}

fn _get_add_time_ids(/*original_size: i32, crops_coords_top_left: i32, target_size: i32, dtype: Kind*/) -> Tensor {
    let add_time_ids = vec![1024, 1024, 0, 0, 1024, 1024];
    // let addition_time_embed_dim = 256;
    // let projection_dim = 1280;
    // let passed_add_embed_dim = add_time_ids.len() * addition_time_embed_dim + projection_dim;
    let t = Tensor::from_slice(&add_time_ids);
    return t.to_kind(Kind::Float).reshape(vec![1, -1]);
    // add_time_ids = list(original_size + crops_coords_top_left + target_size)

    //     passed_add_embed_dim = (
    //         self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
    //     )
    //     expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

    //     if expected_add_embed_dim != passed_add_embed_dim:
    //         raise ValueError(
    //             f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
    //         )

    //     add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    //     return add_time_ids
    // todo!()
}

impl UNet2DConditionModel {
    pub fn forward(&self, xs: &Tensor, timestep: f64, encoder_hidden_states: &Tensor, add_text_embeds: Option<Tensor>) -> Tensor {
        self.forward_with_additional_residuals(xs, timestep, encoder_hidden_states, None, None, add_text_embeds)
    }

    pub fn forward_with_additional_residuals(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        down_block_additional_residuals: Option<&[Tensor]>,
        mid_block_additional_residual: Option<&Tensor>,
        pooled_prompt_embeds: Option<Tensor>
    ) -> Tensor {
        let (bsize, _channels, height, width) = xs.size4().unwrap();
        let device = xs.device();
        let n_blocks = self.config.blocks.len();
        let num_upsamplers = n_blocks - 1;
        let default_overall_up_factor = 2i64.pow(num_upsamplers as u32);
        let forward_upsample_size =
            height % default_overall_up_factor != 0 || width % default_overall_up_factor != 0;
        // 0. center input if necessary
        let xs = if self.config.center_input_sample { xs * 2.0 - 1.0 } else { xs.shallow_clone() };
        // 1. time
        let mut emb = (Tensor::ones([bsize], (xs.kind(), device)) * timestep).apply(&self.time_proj).apply(&self.time_embedding);
        match &self.add_embedding {
            Some(add_embedding) => {
                println!("--->>>>1 {:?} \n{:?} \n{:?}", emb, add_embedding, encoder_hidden_states);
                let text_embeds = pooled_prompt_embeds.unwrap();
                let time_ids = Tensor::concat(&[_get_add_time_ids(), _get_add_time_ids()], 0).to_device(text_embeds.device());
                println!("---->>>> 1.1 {}", time_ids);
                match &self.add_time_proj {
                    Some(add_time_proj) => {
                        let time_embeds = add_time_proj.forward(&time_ids.flatten(0, -1));
                        println!("---->>>>1.5 {}", time_embeds);
                        let text_embeds_size = text_embeds.size();
                        let time_embeds = time_embeds.reshape(vec![text_embeds_size.get(0).unwrap().clone(), -1]);
                        println!("--->>>2 {} \n{}", text_embeds, time_embeds);
                        let add_embeds = Tensor::concat(&[text_embeds, time_embeds], -1);
                        let add_embeds = add_embeds.to_kind(emb.kind());
                        println!("--->>>>3 {}", add_embeds);
                        let aug_emb = add_embedding.forward(&add_embeds);
                        println!("--->>>>4 {}", aug_emb);
                        emb = emb + aug_emb;
                    },
                    None => todo!(),
                }

                // let add_embeds = Tensor::concat()
                // let aug_emb = add_embedding.forward(encoder_hidden_states);
                // println!("--->>>>2");
                // emb = Tensor::concat(&[emb, aug_emb], 0);
            },
            None => {},
        };
        // 2. pre-process
        let xs = xs.apply(&self.conv_in);
        // 3. down
        let mut down_block_res_xs = vec![xs.shallow_clone()];
        let mut xs = xs;
        for down_block in self.down_blocks.iter() {
            let (_xs, res_xs) = match down_block {
                UNetDownBlock::Basic(b) => b.forward(&xs, Some(&emb)),
                UNetDownBlock::CrossAttn(b) => {
                    b.forward(&xs, Some(&emb), Some(encoder_hidden_states))
                }
            };
            down_block_res_xs.extend(res_xs);
            xs = _xs;
        }
        let new_down_block_res_xs =
            if let Some(down_block_additional_residuals) = down_block_additional_residuals {
                let mut v = vec![];
                // A previous version of this code had a bug because of the addition being made
                // in place via += hence modifying the input of the mid block.
                for (i, residuals) in down_block_additional_residuals.iter().enumerate() {
                    v.push(&down_block_res_xs[i] + residuals)
                }
                v
            } else {
                down_block_res_xs
            };
        let mut down_block_res_xs = new_down_block_res_xs;

        // 4. mid
        let xs = self.mid_block.forward(&xs, Some(&emb), Some(encoder_hidden_states));
        let xs = match mid_block_additional_residual {
            None => xs,
            Some(m) => m + xs,
        };
        // 5. up
        let mut xs = xs;
        let mut upsample_size = None;
        for (i, up_block) in self.up_blocks.iter().enumerate() {
            let n_resnets = match up_block {
                UNetUpBlock::Basic(b) => b.resnets.len(),
                UNetUpBlock::CrossAttn(b) => b.upblock.resnets.len(),
            };
            let res_xs = down_block_res_xs.split_off(down_block_res_xs.len() - n_resnets);
            if i < n_blocks - 1 && forward_upsample_size {
                let (_, _, h, w) = down_block_res_xs.last().unwrap().size4().unwrap();
                upsample_size = Some((h, w))
            }
            xs = match up_block {
                UNetUpBlock::Basic(b) => b.forward(&xs, &res_xs, Some(&emb), upsample_size),
                UNetUpBlock::CrossAttn(b) => {
                    b.forward(&xs, &res_xs, Some(&emb), upsample_size, Some(encoder_hidden_states))
                }
            };
        }
        // 6. post-process
        xs.apply(&self.conv_norm_out).silu().apply(&self.conv_out)
    }
}
