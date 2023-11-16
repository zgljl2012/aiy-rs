use tch::{nn::Module, Tensor};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    QuickGelu,
    Gelu,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self {
            Activation::QuickGelu => xs * (xs * 1.702).sigmoid(),
            Activation::Gelu => xs.gelu("none"),
        }
    }
}
