
use aiy_rs::{unet::unet_2d, model_kind::ModelKind};
use tch::Device;

fn main() {
    // let unet_weights = "data/sdxl-base-0.9-unet.safetensors";
    // let mut vs_unet = tch::nn::VarStore::new(Device::cuda_if_available());
    // unet_2d::UNet2DConditionModel::new(vs_unet.root(), 4, 4, ModelKind::SDXL_0_9.unet_config(), &ModelKind::SDXL_0_9);
    // vs_unet.load(unet_weights).unwrap();
}
