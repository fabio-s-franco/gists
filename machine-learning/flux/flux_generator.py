# Author: Fábio Franco (https://github.com/fabio-s-franco)
# Version: 1.0
# Disclaimer: This script is should be used at your own risk. It is provided as is and may not be suitable for your use case.
#             It is the result of a highly experimental project motivated by curiosity. It was done in less than a day and 
#             I have little to no experience with python or marchine-learning techniques. Take the advices here with a pinch of salt.
#             Information here may not be accurate or up-to-date. I am not responsible for any damage or loss caused by this script.
# License: MIT
# Description: Generates images using the FLUX models (schnell and dev) from Black Forest Labs on consumer grade hardware (NVIDIA GPU)
# Assumptions: 
#   - PyTorch >= 2.0 (dismisses xformers)
#   - CUDA >= 11.0 (Ampere and later GPUs)
#   - VRAM >= 8 GB (16 GB Shared Video RAM)
#   - CPU >= 10 cores
#   - RAM >= 32 GB
# Dependencies (beyond project dependencies):
#   - flux local installation (quick-start): https://github.com/black-forest-labs/flux?tab=readme-ov-file#local-installation
#   - diffusers https://github.com/black-forest-labs/flux?tab=readme-ov-file#diffusers-integration
#   - optimum-quanto https://huggingface.co/docs/diffusers/v0.30.0/en/api/pipelines/flux#single-file-loading-for-the-fluxtransformer2dmodel
#   - accelerate # https://huggingface.co/docs/diffusers/en/tutorials/fast_diffusion
# References: 
#   - Flux Docs (Diffusers) - https://huggingface.co/docs/diffusers/en/index
#   - CUDA (WSL) - https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#wsl
# Environment (tested):
#   - CPU(s): 20 (Intel(R) Core(TM) i9-12900H CPU @ 2.90GHz) - 10 cores available for WSL2
#   - GPU(s): NVIDIA GeForce RTX 3080 Laptop GPU (8 GB VRAM) + Intel(R) UHD Graphics (16 GB Shared Video RAM)
#   - RAM: 32 GB DODIMM 3200 MT/s (24 GB for WSL2 and 32 GB swap)
#   - Hard-drive: High-speed NVMe SSD
#   - Host: Windows 11 (10.0.22635.4010)
#   - Virtualization Environment: WSL2 (2.3.17.0 Kernel: 5.15.153.1-2)
#   - Distro: Kali GNU/Linux Rolling (2024.3) x86_64
#   - Python 3.12.4
# Benchmark with default parameters:
#   - Model: dev (40 minutes for 20 steps)
#   - Model: schnell (10 minutes for 4 steps)
# Quick-start:
#   1 - https://github.com/black-forest-labs/flux?tab=readme-ov-file#local-installation
#   2 - Install extra packages
#   ```shell
#   cd $HOME/flux # or any other directory you chose in step 1
#   pip install git+https://github.com/huggingface/diffusers.git
#   pip install optimum-quanto
#   pip install -U transformers accelerate peft
#   ```
#   3 - Run this script inside flux directory (python flux_generator.py --model dev "A dirty beggar wearing old rags, sitting on the sidewalk with its back touching a graffiti wall and a street dog laying by his right side. An RGB lit gaming laptop lays on his lap. To his left, laying against the wall, a sign that reads 'My laptop cannot run a transformer'. Next to the sign a shallow box with several tossed coins.")
#
# Tips for WSL2 users:
#   Nvidia documentation is not very clear about the installation process for WSL. And there were hiccups
#   It may be misleading in scenarios like mine (with Kali Linux) which is not officially supported by Nvidia
#   The imporant thing is:
#   - DON'T install the NVIDIA driver for Linux
#   - IF debian based, use WSL-Ubuntu CUDA Toolkit installation: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu
#   - IF not debian based, read the documnentation carefully, as the CUDA toolkit may overwrite the drivers installed by Windows for WSL2
#   - The windows driver installation also installs the WSL2 driver, so don't worry about it
#   - Run the command `nvidia-smi` to check if the driver is installed correctly (not nvidia-detect as in their docs)
#   - GPU Memory usage will show as N/A in WSL2, because the controller is in Windows (can be viewed in Task Manager or Resource Monitor)
# 
# TODO: (for the unknown future)
#   - https://huggingface.co/docs/diffusers/main/en/optimization/fp16#tiny-autoencoder
#   - Attempt more up to date tokenizers (openai/clip-vit-base-patch32 and https://huggingface.co/google/siglip-so400m-patch14-384 lead to runtime errors)

import torch
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast


def generate(prompt: str, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", offload: bool = True, steps: int = -1, width: int = 1024, height: int = 1024, output: str = None, cuda_pipeline: bool = False):
    model_id = f"black-forest-labs/FLUX.1-{model_name}"
    max_sequence_length = 256 if model_name == "schnell" else 512

    ### For Ampere and later GPUs, this will speed up the model (https://huggingface.co/docs/diffusers/en/optimization/fp16#tensorfloat-32). Otherwise # dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.float16
    ###

    # https://huggingface.co/docs/diffusers/v0.30.0/en/api/pipelines/flux#single-file-loading-for-the-fluxtransformer2dmodel
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", use_safetensors=True)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", clean_up_tokenization_spaces=True)
    text_encoder_2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True)
    tokenizer_2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2", torch_dtype=dtype, use_safetensors=True)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype, use_safetensors=True)
    transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype, use_safetensors=True)

    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer
    )

    ## The statements below only work when pipe.to('cuda') is called, but I could not get it to stay within memory limits (8 GB VRAM + 16 GB Shared Video RAM)
    if cuda_pipeline:
        pipe = pipe.to("cuda")
        pipe.enable_vae_slicing() # https://huggingface.co/docs/diffusers/en/optimization/memory#sliced-vae
        pipe.enable_vae_tiling() # https://huggingface.co/docs/diffusers/en/optimization/memory#tiled-vae
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True) # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0#torchcompile
    else:
        pipe.enable_model_cpu_offload() if offload else None # saves VRAM https://huggingface.co/docs/diffusers/en/optimization/memory#model-offloading
        # pipe.enable_sequential_cpu_offload() # Couldn't get this to work properly, saves more VRAM but apparently also makes inference even slower: https://huggingface.co/docs/diffusers/en/optimization/memory#cpu-offloading
    
    with torch.inference_mode():
        image = pipe(
            prompt,
            height=width,
            width=height,
            output_type="pil",
            num_inference_steps=steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator(device)
        ).images[0]
        
        output_file_name = output if output else f"flux-{model_name}.png"
        image.save(output_file_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux Prompt to Image Generator")
    parser.add_argument("prompt", type=str, help="The prompt to generate the image", metavar="\"<PROMPT>\"")
    parser.add_argument("--model", type=str, default="schnell", choices=["schnell", "dev"], help="Which Flux model to use", required=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Device to use for inference (default: cuda - better performance)", required=False)
    parser.add_argument("--offload_off", action="store_true", help="Disables model offload to CPU and uses the GPU as needed (default: offload is enabled). Lowers VRAM usage.", required=False)
    parser.add_argument("--steps", type=int, default=0, help="Number of inference steps. Higher, yields better image quality and is proportionally slower. Default 4 for schnell and 20 for dev", required=False)
    parser.add_argument("--width", type=int, default=1024, help="Output image Width (default: 1024)", required=False)
    parser.add_argument("--height", type=int, default=1024, help="Output image Height (default: 1024)", required=False)
    parser.add_argument("--output", type=str, help="Output file name (default: flux-<model_name>.png)", required=False)
    parser.add_argument("--cuda_pipeline", action="store_true", help="Enables CUDA for loading the entire pipeline, should perform better, but uses more VRAM. (default: cpu - lower VRAM usage)", required=False)

    args = parser.parse_args()

    offload = not args.offload_off
    steps = args.steps if args.steps > 0 else 4 if args.model == "schnell" else 20

    generate(args.prompt, args.model, args.device, offload, steps, args.width, args.height, args.output, args.cuda_pipeline)
