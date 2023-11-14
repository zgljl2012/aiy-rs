use super::AiyStableDiffusion;


pub trait AiySdBuilder {
    fn from_single_weights_file(bpe_file: &str, safetensors_file_path: &str) -> anyhow::Result<AiyStableDiffusion>;
}
