from diffusers import DiffusionPipeline
import torch

TOKEN = "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"

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

image = pipe(prompt=prompt, output_type="latent", num_inference_steps=20).images

# image.save("sd_0_original.png")

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

images = pipe(prompt=prompt, image=image, num_inference_steps=10).images
images[0].save("sd_0_refined.png")
