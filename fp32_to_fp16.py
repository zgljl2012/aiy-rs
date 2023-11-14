import argparse
import os
from collections import defaultdict

import torch

from safetensors.torch import load_file, save_file

def convert_file(
    file: str,
    target: str,
):
    # loaded = torch.load(pt_filename, map_location="cpu")
    # if "state_dict" in loaded:
    #     loaded = loaded["state_dict"]
    # shared = shared_pointers(loaded)
    # for shared_weights in shared:
    #     for name in shared_weights[1:]:
    #         loaded.pop(name)

    # # For tensors to be contiguous
    # loaded = {k: v.contiguous() for k, v in loaded.items()}

    # dirname = os.path.dirname(sf_filename)
    # os.makedirs(dirname, exist_ok=True)
    # save_file(loaded, sf_filename, metadata={"format": "pt"})
    # check_file_size(sf_filename, pt_filename)
    reloaded = load_file(file)
    updated = { k: v.half() for k, v in reloaded.items()}
    save_file(updated, target, metadata={"format": "pt"})

if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert f32 to f16.
    It is PyTorch exclusive for now.
    It converts them locally.

    python fp32_to_fp16.py --dist "D:\\stable diffusion\\stable-diffusion-webui-1.6.0\\models\\Stable-diffusion\\sd_xl_base_0.9.f16.safetensors" "D:\\stable diffusion\\stable-diffusion-webui-1.6.0\\models\\Stable-diffusion\\sd_xl_base_0.9.safetensors"
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--dist",
        type=str,
        required=True,
        help="target file",
    )
    parser.add_argument(
        "file",
        type=str,
        help="Weights file",
    )
    args = parser.parse_args()
    file = args.file
    dist = args.dist
    print(dist)
    if os.path.exists(dist):
        print(f'Error: {dist} already exists')
    else:
        convert_file(file, dist)
        print('Success')
