use std::path::Path;

use tch::{Device, Kind, Tensor};

pub(crate) fn file_open<P: AsRef<Path>>(path: P) -> anyhow::Result<std::fs::File> {
    std::fs::File::open(path.as_ref()).map_err(|e| {
        let context = format!("error opening {:?}", path.as_ref().to_string_lossy());
        anyhow::Error::new(e).context(context)
    })
}

pub fn get_device() -> Device {
    let accelerator_device =
            if tch::utils::has_mps() { Device::Mps } else { Device::cuda_if_available() };

    return accelerator_device
}

pub fn output_filename(
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

pub fn has_nan(xs: &Tensor) -> bool {
    let num = xs.isnan().to_kind(Kind::Int).sum(Kind::Int).greater(0).to_string(10).unwrap();
    num.starts_with("1")
}

pub fn count_nan(xs: &Tensor) -> String {
    let num = xs.isnan().to_kind(Kind::Int).sum(Kind::Int).to_string(10).unwrap();
    num
}
