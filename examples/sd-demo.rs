use std::time::SystemTime;

use aiy_rs::{
    aiy_sd::AiyStableDiffusion,
    example_models::{sdxl_0_9, bra_v7},
};

fn run() -> anyhow::Result<()> {
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    println!("MPS available: {}", tch::utils::has_mps());

    let start = SystemTime::now();

    let aiy = AiyStableDiffusion::new(bra_v7())
    .unwrap();

    println!(
        "=== Device setup: {:?}",
        SystemTime::now().duration_since(start).unwrap()
    );

    aiy.text_2_image(
        // "A very realistic photo of a rusty robot walking on a sandy beach, sea and a green tree",
        // "A high-resolution photograph of a waterfall in autumn; muted tone",
        // "An astronaut riding a green horse",
        // "",
        "ultra realistic close up portrait ((beautiful pale cyberpunk female with heavy black eyeliner)), blue eyes, shaved side haircut, hyper detail, cinematic lighting, magic neon, dark red city, Canon EOS R3, her is touch her hair, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame, 8K",
        "painting, extra fingers, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
        "./sd_final.png",
        false,
        20,
        1,
        32,
        None,
        None
    )?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    run()
}
