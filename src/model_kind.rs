use tch::nn;

#[derive(Debug, Clone)]
pub enum ModelKind {
    SD1_5,
    SD2_1,
}

pub struct VaeAttensionPaths<'a> {
    pub group_norm: nn::Path<'a>,
    pub query: nn::Path<'a>,
    pub key: nn::Path<'a>,
    pub value: nn::Path<'a>,
    pub proj_attn: nn::Path<'a>,
}

impl ModelKind {
    pub fn vae_attension_paths<'a>(&self, v: nn::Path<'a>) -> VaeAttensionPaths<'a> {
        let v1_5: VaeAttensionPaths<'a> = VaeAttensionPaths {
            group_norm: &v / "group_norm",
            query: &v / "to_q",
            key: &v / "to_k",
            value: &v / "to_v",
            proj_attn: &v / "to_out" / 0,
        };
        let v2_1: VaeAttensionPaths<'a> = VaeAttensionPaths {
            group_norm: &v / "group_norm",
            query: &v / "query",
            key: &v / "key",
            value: &v / "value",
            proj_attn: &v / "proj_attn",
        };
        match &self {
            ModelKind::SD1_5 => v1_5,
            ModelKind::SD2_1 => v2_1,
        }
    }
}
