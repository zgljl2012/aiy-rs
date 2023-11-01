use std::collections::HashSet;

use tch::Tensor;

fn main() {
    // let path = "data/clip_v2.1.safetensors";
    // let path = "data/unet_v2.1.safetensors"; // {"mid_block", "down_blocks", "conv_in", "conv_out", "conv_norm_out", "time_embedding", "up_blocks"}
    // let path = "D:\\stable diffusion\\stable-diffusion-webui-1.6.0\\models\\Stable-diffusion\\sd_xl_base_1.0.safetensors";
    // LEVEL1: {"model", "cond_stage_model", "first_stage_model"}
    let path = "D:\\stable diffusion\\stable-diffusion-webui-1.6.0\\models\\Stable-diffusion\\v2-1_768-nonema-pruned.safetensors";
    let mut first_level = HashSet::new();
    let t = Tensor::read_safetensors(path).unwrap();
    for (name, _value) in &t {
        if name.starts_with("model.diffusion_model") {
            let fields: Vec<String> = name.clone().split(".").into_iter().map(|v| v.to_owned()).collect();
            if fields.len() > 2 {
                first_level.insert(fields[2].clone());
            }
        }
        if name.starts_with("first_stage_model.text") || name.starts_with("text_model") {
            println!("{}", name);
        }
    }
    println!("{:?}", first_level);
}
