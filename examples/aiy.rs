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

    // 如果指定了此参数，则为图生图
    #[arg(long)]
    with_input_image: Option<String>,

    #[arg(long, default_value = "0.8")]
    strength: f64,

    // 如果指定了此参数，就是指定了 refine-model，需先文生图，再图生图
    #[arg(long)]
    with_refine_model: Option<String>,

    #[arg(long, default_value = "15")]
    refine_steps: usize,

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
        with_input_image,
        strength,
        with_refine_model,
        refine_steps,
    } = args;
    let start = SystemTime::now();
    let builder = AiySdBuilder::new(&aiy_home);
    let aiy = builder.from_repo(&model).unwrap();
    println!(
        "=== Device setup: {:?}",
        SystemTime::now().duration_since(start).unwrap()
    );
    match with_input_image {
        Some(input_image) => {
            println!(">> image-to-image");
            aiy.image_2_image(
                &input_image,
                &prompt,
                &negative_prompt,
                &output,
                false,
                n_steps,
                1,
                seed as i64,
                Some(strength),
            )
            .unwrap();
        }
        None => {
            println!(">> text-to-image");
            aiy.text_2_image(
                &prompt,
                &negative_prompt,
                &output,
                false,
                n_steps,
                1,
                seed as i64,
                if width > 0 { Some(width) } else { None },
                if height > 0 { Some(height) } else { None },
            )
            .unwrap();
            match with_refine_model {
                Some(_model) => {
                    // 效果不好（代码中对 refine 模型的支持不对，但没有时间改了），暂不实现
                    todo!()
                    // let name = output.trim_end_matches(".png");
                    // let refine_output = format!("{name}_refine.png");
                    // println!(">> image-to-image with refine model {model}");
                    // let builder = AiySdBuilder::new(&aiy_home);
                    // let aiy = builder.from_repo(&model).unwrap();
                    // aiy.image_2_image(
                    //     &output,
                    //     &prompt,
                    //     &negative_prompt,
                    //     &refine_output,
                    //     false,
                    //     refine_steps,
                    //     1,
                    //     seed as i64,
                    //     Some(strength),
                    // )
                    // .unwrap();
                }
                None => {}
            }
        }
    };
}
