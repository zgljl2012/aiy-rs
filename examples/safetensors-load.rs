use std::collections::HashSet;

use tch::Tensor;

/*
clip_v2.1:
    text_model
        {"encoder", "final_layer_norm", "embeddings"}

v2-1_768-nonema-pruned.safetensors:
    first_stage_model:
        {"decoder", "encoder", "quant_conv", "post_quant_conv"}
    model:
        diffusion_model:
            {"output_blocks", "out", "middle_block", "input_blocks", "time_embed"}
    cond_stage_model:
        model:
            {"text_projection", "positional_embedding", "logit_scale", "token_embedding", "ln_final", "transformer"}

sd_xl_base_1.0.safetensors:
    {"conditioner", "first_stage_model", "model"}
*/

fn main() {
    // let path = "data/clip_v2.1.safetensors";
    // let path = "data/vae_v2.1.fp16.safetensors";
    let path = "data/sdxl-base-0.9-unet.safetensors";
    // let path = "data/unet_v2.1.fp16.safetensors";
    // let path = "data/unet_v2.1.safetensors"; // {"mid_block", "down_blocks", "conv_in", "conv_out", "conv_norm_out", "time_embedding", "up_blocks"}
    // let path = "D:\\stable diffusion\\stable-diffusion-webui-1.6.0\\models\\Stable-diffusion\\sd_xl_base_1.0.safetensors";
    // let path = "D:\\stable diffusion\\stable-diffusion-webui-1.6.0\\models\\Stable-diffusion\\v2-1_768-nonema-pruned.safetensors";
    // LEVEL1: {"model", "cond_stage_model", "first_stage_model"}
    // let path = "D:\\stable diffusion\\stable-diffusion-webui-1.6.0\\models\\Stable-diffusion\\v2-1_768-nonema-pruned.safetensors";
    let mut first_level = HashSet::new();
    let t = Tensor::read_safetensors(path).unwrap();
    for (name, _value) in &t {
        if name.starts_with("up_blocks") {
            println!("{}", name);
            let fields: Vec<String> = name.clone().split(".").into_iter().map(|v| v.to_owned()).collect();
            if fields.len() > 2 {
                first_level.insert(fields[0].clone());
            }
        }
        // if name.starts_with("first_stage_model.text") || name.starts_with("text_model") {
        //     println!("{}", name);
        // }
    }
    // println!("{:?}", first_level);
}
