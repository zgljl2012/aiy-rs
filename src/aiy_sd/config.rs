use std::fs;

use crate::model_kind::ModelKind;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiyBaseConfig {
    pub default_width: usize,
    pub default_height: usize,
    pub model_kind: ModelKind
}

impl AiyBaseConfig {
    pub fn from_file<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<Self> {
        let file = fs::read_to_string(path)?;
        let cfg: AiyBaseConfig = toml::from_str(&file)?;
        Ok(cfg)
    }
}
