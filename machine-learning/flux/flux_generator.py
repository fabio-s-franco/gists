# Author: Fábio Franco (https://github.com/fabio-s-franco)
# Version: 1.1
# Disclaimer: This script is should be used at your own risk. It is provided as is and may not be suitable for your use case.
#             It is the result of a highly experimental project motivated by curiosity. It was done in less than a day and 
#             I have little to no experience with python or marchine-learning techniques. Take the advices here with a pinch of salt.
#             Information here may not be accurate or up-to-date. I am not responsible for any damage or loss caused by this script.
# License: MIT
# Description: Generates images using the FLUX models (schnell and dev) from Black Forest Labs on consumer grade hardware (NVIDIA GPU)

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

    tokenizerModel = "openai/clip-vit-large-patch14"

    tokenizer = CLIPTokenizer.from_pretrained(tokenizerModel, clean_up_tokenization_spaces=True) # Currently, the default when clean_up_tokenization_spaces is not passed is False, but the behavior has been deprecated and will change.
   
    # How many tokens this model can handle
    max_tokens = tokenizer.model_max_length

    tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    num_tokens = len(tokens)
    
    if num_tokens > max_tokens:
        excess_tokens = tokens[max_tokens:]
        excess_text = tokenizer.decode(excess_tokens, clean_up_tokenization_spaces=True)
        raise ValueError(f"The maximum amount of tokens is {max_tokens}, your prompt contains {num_tokens}, which is exceeded with the text: '{excess_text}'")

    # https://huggingface.co/docs/diffusers/v0.30.0/en/api/pipelines/flux#single-file-loading-for-the-fluxtransformer2dmodel
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", use_safetensors=True)
    text_encoder = CLIPTextModel.from_pretrained(tokenizerModel)
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
    parser.add_argument("--offload_off", action="store_true", required=False,
        help="""When enabled, model offloading utilizes spare CPU capacity, and only uses the GPU as the need arises, 
        therefore conserves GPU VRAM usage. Passing this flag disables model offloading (default: offload is enabled). 
        This flag improves performance, but only use this flag if you have enough GPU power and memory, otherwise you 
        will get an OutOfMemory error."""
    )
    parser.add_argument("--steps", type=int, default=0, help="Number of inference steps. Higher, yields better image quality and is proportionally slower. Default 4 for schnell and 20 for dev", required=False)
    parser.add_argument("--width", type=int, default=1024, help="Output image Width (default: 1024)", required=False)
    parser.add_argument("--height", type=int, default=1024, help="Output image Height (default: 1024)", required=False)
    parser.add_argument("--output", type=str, help="Output file name (default: flux-<model_name>.png)", required=False)
    parser.add_argument("--cuda_pipeline", action="store_true", help="Enables CUDA for loading the entire pipeline, should perform better, but uses more VRAM. (default: cpu - lower VRAM usage)", required=False)

    args = parser.parse_args()

    offload = not args.offload_off
    steps = args.steps if args.steps > 0 else 4 if args.model == "schnell" else 20

    try:
        generate(args.prompt, args.model, args.device, offload, steps, args.width, args.height, args.output, args.cuda_pipeline)
    except ValueError as e:
        print(e)
        exit(1)
