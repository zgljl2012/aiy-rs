use std::time::SystemTime;

use aiy_rs::aiy_sd::AiySdBuilder;
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    #[arg(long, default_value = "")]
    negative_prompt: String,

    #[arg(long, default_value = "sdxl-base-1.0")]
    model: String,

    #[arg(long, default_value = "E://.aiy-repos")]
    aiy_home: String,

    #[arg(long, default_value = "./sd_final.png")]
    output: String,

    #[arg(long, default_value = "10")]
    n_steps: usize,

    #[arg(long, default_value = "9527")]
    seed: usize,

    #[arg(long, default_value = "0")]
    width: usize,

    #[arg(long, default_value = "0")]
    height: usize,
}

fn main() {
    let args = Args::parse();
    let Args {
        prompt,
        n_steps,
        seed,
        negative_prompt,
        aiy_home,
        model,
        width,
        height,
        output,
    } = args;
    let start = SystemTime::now();
    let builder = AiySdBuilder::new(&aiy_home);
    let aiy = builder.from_repo(&model).unwrap();
    println!(
        "=== Device setup: {:?}",
        SystemTime::now().duration_since(start).unwrap()
    );
    aiy.text_2_image(
            &prompt,
            &negative_prompt,
            &output,
            false,
            n_steps,
            1,
            seed as i64,
            if width > 0 { Some(width) } else { None },
            if height > 0 { Some(height) } else { None }
        ).unwrap();
}
