use crate::clip::Config;


pub enum ClipConfig {
    V1_5,
    V2_1
}

impl ClipConfig {
    pub fn config(&self) -> Config {
        match self {
            ClipConfig::V1_5 => ClipConfig::v1_5(),
            ClipConfig::V2_1 => ClipConfig::v2_1(),
        }
    }

    fn v1_5() -> Config {
        Config::v1_5()
    }

    fn v2_1() -> Config {
        Config::v2_1()
    }
}
