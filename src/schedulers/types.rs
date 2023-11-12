use tch::Tensor;


pub trait Scheduler {
    fn timesteps(&self) -> Vec<f64>;
    fn scale_model_input(&self, sample: Tensor, timestep: f64) -> Tensor;
    fn step(&self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Tensor;
    fn init_noise_sigma(&self) -> f64;
    fn add_noise(&self, original_samples: &Tensor, noise: Tensor, timestep: f64) -> Tensor;
}
