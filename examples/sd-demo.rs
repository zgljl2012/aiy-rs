use std::time::SystemTime;

use aiy_rs::{aiy_sd::{AiyStableDiffusion, AiyConfig}, clip_configs::ClipConfig};
// Stable Diffusion implementation inspired:
// - Huggingface's amazing diffuser Python api: https://huggingface.co/blog/annotated-diffusion
// - Huggingface's (also amazing) blog post: https://huggingface.co/blog/annotated-diffusion
// - The "Grokking Stable Diffusion" notebook by Jonathan Whitaker.
// https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing
//
// In order to run this, the weights first have to be downloaded and converted by following
// the instructions below.
//
// mkdir -p data && cd data
// wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
// gunzip bpe_simple_vocab_16e6.txt.gz
//
// Getting the weights then depend on the stable diffusion version (1.5 or 2.1).
//
// # How to get the weights for Stable Diffusion 2.1.
//
// 1. Clip Encoding Weights
// wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/fp16/text_encoder/pytorch_model.bin -O clip.bin
// From python, extract the weights and save them as a .npz file.
//   import torch
//   from safetensors.torch import save_file
//
//   model = torch.load("./clip.bin")
//   save_file("./clip_v2.1.safetensors", **{k: v.numpy() for k, v in model.items() if "text_model" in k})
//
// 2. VAE and Unet Weights
// wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/fp16/vae/diffusion_pytorch_model.bin -O vae.bin
// wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/fp16/unet/diffusion_pytorch_model.bin -O unet.bin
//
//   import torch
//   from safetensors.torch import save_file
//   model = torch.load("./vae.bin")
//   save_file(dict(model), './vae.safetensors')
//   model = torch.load("./unet.bin")
//   save_file(dict(model), './unet.safetensors')
//
// # How to get the weights for Stable Diffusion 1.5.
//
// 1. Clip Encoding Weights
// wget https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin
// From python, extract the weights and save them as a .npz file.
//   import torch
//   from safetensors.torch import save_file
//
//   model = torch.load("./pytorch_model.bin")
//   save_file("./pytorch_model.safetensors", **{k: v.numpy() for k, v in model.items() if "text_model" in k})
//
// 2. VAE and Unet Weights
// https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/diffusion_pytorch_model.bin
// https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/diffusion_pytorch_model.bin
//
//   import torch
//   from safetensors.torch import save_file
//   model = torch.load("./vae.bin")
//   save_file(dict(model), './vae.safetensors')
//   model = torch.load("./unet.bin")
//   save_file(dict(model), './unet.safetensors')
use clap::Parser;
use diffusers::pipelines::stable_diffusion;
use tch::{Device, Kind, Tensor};

const GUIDANCE_SCALE: f64 = 7.5;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    /// When set, use the CPU for the listed devices, can be 'all', 'unet', 'clip', etc.
    /// Multiple values can be set.
    #[arg(long)]
    cpu: Vec<String>,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<i64>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<i64>,

    /// The UNet weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    #[arg(long, value_name = "FILE", default_value = "data/bpe_simple_vocab_16e6.txt")]
    /// The file specifying the vocabulary to used for tokenization.
    vocab_file: String,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    #[arg(long)]
    sliced_attention_size: Option<i64>,

    /// The number of steps to run the diffusion for.
    #[arg(long, default_value_t = 30)]
    n_steps: usize,

    /// The random seed to be used for the generation.
    #[arg(long, default_value_t = 32)]
    seed: i64,

    /// The number of samples to generate.
    #[arg(long, default_value_t = 1)]
    num_samples: i64,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,

    /// Use autocast (disabled by default as it may use more memory in some cases).
    #[arg(long, action)]
    autocast: bool,

    #[arg(long, value_enum, default_value = "v2-1")]
    sd_version: StableDiffusionVersion,

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum StableDiffusionVersion {
    V1_5,
    V2_1,
}

impl Args {
    fn unet_weights(&self) -> String {
        match &self.unet_weights {
            Some(w) => w.clone(),
            None => match self.sd_version {
                StableDiffusionVersion::V1_5 => "data/unet.safetensors".to_string(),
                StableDiffusionVersion::V2_1 => "data/unet_v2.1.safetensors".to_string(),
            },
        }
    }
}

fn output_filename(
    basename: &str,
    sample_idx: i64,
    num_samples: i64,
    timestep_idx: Option<usize>,
) -> String {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    } else {
        basename.to_string()
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}

fn run(args: Args) -> anyhow::Result<()> {
    let unet_weights = args.unet_weights();
    let Args {
        prompt,
        cpu,
        height,
        width,
        n_steps,
        seed,
        vocab_file,
        final_image,
        sliced_attention_size,
        num_samples,
        sd_version,
        ..
    } = args;
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    println!("MPS available: {}", tch::utils::has_mps());

    let sd_config = match sd_version {
        StableDiffusionVersion::V1_5 => {
            stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V2_1 => {
            stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
        }
    };

    let start = SystemTime::now();

    // 确定哪个模型使用 CPU，默认都不用
    let device_setup = diffusers::utils::DeviceSetup::new(cpu);
    let unet_device = device_setup.get("unet");
    let scheduler = sd_config.build_scheduler(n_steps);

    let aiy = AiyStableDiffusion::new(AiyConfig {
        vocab_path: vocab_file.clone(),
        clip_weights_path: "data/clip_v2.1.safetensors".to_string(),
        clip_config: ClipConfig::V2_1.config(),
        // VAE
        vae_weights_path: "data/vae_v2.1.safetensors".to_string()
    }).unwrap();

    println!("=== Device setup: {:?}", SystemTime::now().duration_since(start).unwrap());

    let text_embeddings = aiy.embed_prompts(&prompt, "sand orange human sky, water, sea")?;

    let no_grad_guard = tch::no_grad_guard();

    println!("Building the unet.");
    let unet = sd_config.build_unet(&unet_weights, unet_device, 4)?;

    let bsize = 1;
    let start = SystemTime::now();
    for idx in 0..num_samples {
        tch::manual_seed(seed + idx);
        let mut latents = Tensor::randn(
            [bsize, 4, sd_config.height / 8, sd_config.width / 8],
            (Kind::Float, unet_device),
        );

        // scale the initial noise by the standard deviation required by the scheduler
        latents *= scheduler.init_noise_sigma();

        for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
            println!("Timestep {timestep_index}/{n_steps}");
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0);

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep);
            let noise_pred = unet.forward(&latent_model_input, timestep as f64, &text_embeddings);
            let noise_pred = noise_pred.chunk(2, 0);
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            let noise_pred =
                noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE; 
            latents = scheduler.step(&noise_pred, timestep, &latents);

            // 生成中间过程图片
            if args.intermediary_images {
                let latents = latents.to(aiy.vae_device);
                let image = aiy.vae_decode(&latents);
                let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
                let image = (image * 255.).to_kind(Kind::Uint8);
                let final_image =
                    output_filename(&final_image, idx + 1, num_samples, Some(timestep_index + 1));
                tch::vision::image::save(&image, final_image)?;
            }
        }

        println!("Generating the final image for sample {}/{}.", idx + 1, num_samples);
        let latents = latents.to(aiy.vae_device);
        let image = aiy.vae_decode(&latents);
        let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
        let image = (image * 255.).to_kind(Kind::Uint8);
        let final_image = output_filename(&final_image, idx + 1, num_samples, None);
        tch::vision::image::save(&image, final_image)?;
    }
    println!("=== Generated image: {:?}", SystemTime::now().duration_since(start).unwrap());
    drop(no_grad_guard);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if !args.autocast {
        run(args)
    } else {
        tch::autocast(true, || run(args))
    }
}
