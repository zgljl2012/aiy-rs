//! # Variational Auto-Encoder (VAE) Models.
//!
//! Auto-encoder models compress their input to a usually smaller latent space
//! before expanding it back to its original shape. This results in the latent values
//! compressing the original information.
use crate::model_kind::ModelKind;
use tch::{nn, Tensor};

use self::{decoder::{Decoder, DecoderConfig}, encoder::{Encoder, EncoderConfig}};

mod decoder;
mod encoder;

#[derive(Debug, Clone)]
pub struct AutoEncoderKLConfig {
    pub block_out_channels: Vec<i64>,
    pub layers_per_block: i64,
    pub latent_channels: i64,
    pub norm_num_groups: i64,
}

impl Default for AutoEncoderKLConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 1,
            latent_channels: 4,
            norm_num_groups: 32,
        }
    }
}

pub struct DiagonalGaussianDistribution {
    mean: Tensor,
    std: Tensor,
    device: tch::Device,
}

impl DiagonalGaussianDistribution {
    pub fn new(parameters: &Tensor) -> Self {
        let mut parameters = parameters.chunk(2, 1).into_iter();
        let mean = parameters.next().unwrap();
        let logvar = parameters.next().unwrap();
        let std = (logvar * 0.5).exp();
        let device = std.device();
        DiagonalGaussianDistribution { mean, std, device }
    }

    pub fn sample(&self) -> Tensor {
        let sample = Tensor::randn_like(&self.mean).to(self.device);
        &self.mean + &self.std * sample
    }
}

// https://github.com/huggingface/diffusers/blob/970e30606c2944e3286f56e8eb6d3dc6d1eb85f7/src/diffusers/models/vae.py#L485
// This implementation is specific to the config used in stable-diffusion-v1-5
// https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json
#[derive(Debug)]
pub struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: nn::Conv2D,
    post_quant_conv: nn::Conv2D,
    pub config: AutoEncoderKLConfig,
}

impl AutoEncoderKL {
    pub fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: AutoEncoderKLConfig,
        base_model: ModelKind
    ) -> Self {
        // println!("----->>>>>999 {}", config.layers_per_block);
        let latent_channels = config.latent_channels;
        let encoder_cfg = EncoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
            double_z: true,
        };
        let encoder = Encoder::new(&vs / "encoder", in_channels, latent_channels, encoder_cfg, base_model.clone());
        let decoder_cfg = DecoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
        };
        let decoder = Decoder::new(&vs / "decoder", latent_channels, out_channels, decoder_cfg, base_model);
        let conv_cfg = Default::default();
        let quant_conv =
            nn::conv2d(&vs / "quant_conv", 2 * latent_channels, 2 * latent_channels, 1, conv_cfg);
        let post_quant_conv =
            nn::conv2d(&vs / "post_quant_conv", latent_channels, latent_channels, 1, conv_cfg);
        Self { encoder, decoder, quant_conv, post_quant_conv, config }
    }

    /// Returns the distribution in the latent space.
    pub fn encode(&self, xs: &Tensor) -> DiagonalGaussianDistribution {
        let parameters = xs.apply(&self.encoder).apply(&self.quant_conv);
        DiagonalGaussianDistribution::new(&parameters)
    }

    /// Takes as input some sampled values.
    pub fn decode(&self, xs: &Tensor) -> Tensor {
        let t = xs.apply(&self.post_quant_conv);
        let t2 = t.apply(&self.decoder);
        t2
    }
}
