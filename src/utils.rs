use std::path::Path;

use tch::Device;

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
