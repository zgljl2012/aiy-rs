use crate::unet_2d;


pub enum UNetConfig {
    V1_5,
    V2_1
}

impl UNetConfig {
    pub fn config(&self) -> unet_2d::UNet2DConditionModelConfig {
        let bc = |out_channels, use_cross_attn, attention_head_dim| unet_2d::BlockConfig {
            out_channels,
            use_cross_attn,
            attention_head_dim,
        };
        // The size of the sliced attention or 0 for automatic slicing (disabled by default)
        let sliced_attention_size = None; 
        // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/config.json
        let v1_5_unet = unet_2d::UNet2DConditionModelConfig {
            blocks: vec![bc(320, true, 8), bc(640, true, 8), bc(1280, true, 8), bc(1280, false, 8)],
            center_input_sample: false,
            cross_attention_dim: 768,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            layers_per_block: 2,
            mid_block_scale_factor: 1.,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size,
            use_linear_projection: false,
        };
        // v2.1
        // https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/unet/config.json
        let v2_1_unet = unet_2d::UNet2DConditionModelConfig {
            blocks: vec![
                bc(320, true, 5),
                bc(640, true, 10),
                bc(1280, true, 20),
                bc(1280, false, 20),
            ],
            center_input_sample: false,
            cross_attention_dim: 1024,
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
        match self {
            UNetConfig::V1_5 => v1_5_unet,
            UNetConfig::V2_1 => v2_1_unet,
        }
    }
}
