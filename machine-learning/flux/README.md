# Flux Text-to-Image Generator (using your home computer)

## Introduction (Shiny samples down below)

This script is a helper to generate images from text using the FLUX open source machine-learning models, developed by [Black Forest Labs](https://blackforestlabs.ai/).
The goal is to be able to generate these images relying only on your local compute resources (no fee, no limits beyond your own hardware) and open source models.
At the time of this writing FLUX is offered in 3 flavors:

- schnell (small) - Free to use under Apache 2.0 license (please confirm licensing details in the source repository).
- dev - A large similar to the paid version, but can only be used for non-commercial purposes (license details in the model's repository)
- pro - The most powerful and only accessible via api. This is a commercial product and at the moment only a select number of partners can use it directly. These partners then offer to third party customers.

This repo is focused on schnell and dev models which you can execute locally on your machine (downloading the dev model, requires registering an account with HF).
I used the [CLI method](https://huggingface.co/docs/hub/en/datasets-polars-auth#cli) as it was very practical to do right from within the Flux repo.

Flux open source models are developed by [Black Forest Labs](https://blackforestlabs.ai/) and allow for consumer-grade hardware (albeit high-end) to generate impressive text-to-image with an NVIDIA GPU using CUDA Toolkit.
I would need to do a bit more research to determine if it is also feasible to do the same with Non NVIDIA GPUs or exclusively with CPUs.

Nevertheles, because I could not get it work with the instructions in flux repo, nor the could successfully run the demo application from the [repository](https://github.com/black-forest-labs/flux), I just simply did not give up until I suceeded.
This repository was not planned, but I decided it was worth the extra effort to make this more tangible, specially because it is really cool and deserves to be more accessible to people like me, that knows nothing about Python and ML algorithms.

I ran into many problems until I got it right, specially out of memory errors (VRAM), so after figuring a way to generate an image within a sane amount of time, I decided to publish my findings and round up a helper script to help others with an unorthodox setup like mine that don't mind tinkering around.

Please visit <https://github.com/black-forest-labs/flux> repository where this work is derived from and made this possible.

## Disclaimer

This script is provided as is and should be used at your own risk. It is the result of a highly experimental project motivated by curiosity and was done in less than a day. I have little to no experience with Python or machine learning techniques. Take the advice provided here with a pinch of salt.

The information provided here may not be accurate or up-to-date. I will be happy to update it, to be more concise and technically and terminologically correct. Feel free to submit a PR.

> **_By utilizing the scripts here provided and all accompanying information in this repository, you accept full liability for any potential damages or losses incurred._**

## Assumptions

Although this is supposed to be executable in "consumer-grade" hardware, the hardware used here is far from low-end. You will still need a relatively powerful hardware (and GPU) to succeed in the manner explained here. However, it may be a starting point for further optimising it to be even easier on compute resources.
I recognize that may failures trying to run this model may simply be due to lack of experience in the field and that a better way may exist to run these models more efficiently or with lower hardware requirements. I could not do so, at this first iteration with it.

You may need to make adjustments to your specific case if your goal is to run this generator at home.
To have a high chance of success, assume the following requirements below as a baseline, as there was very little breathing room (that is RAM and VRAM) left, to run these models:

- PyTorch >= 2.0 (dismisses xformers)
- CUDA >= 11.0 (Ampere and later GPUs)
- VRAM >= 8 GB (16 GB Shared Video RAM)
- CPU >= 10 cores
- RAM >= 32 GB

## Dependencies

In addition to project dependencies, the following dependencies are required:

- [flux](https://github.com/black-forest-labs/flux?tab=readme-ov-file#local-installation) (local installation)
- [diffusers](https://github.com/black-forest-labs/flux?tab=readme-ov-file#diffusers-integration)
- [optimum-quanto](https://huggingface.co/docs/diffusers/v0.30.0/en/api/pipelines/flux#single-file-loading-for-the-fluxtransformer2dmodel)
- [accelerate](https://huggingface.co/docs/diffusers/en/tutorials/fast_diffusion)

## References

- [Flux Docs (Diffusers)](https://huggingface.co/docs/diffusers/en/index)
- [CUDA (WSL)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#wsl)

## My Environment (where tests succeeded)

This was tested on a personal laptop with a setup that is anything but standard:

1. It runs on a virtualized linux environment
2. This linux environment is under WSL2 (which in turn has a Windows Host)
3. You need to build your own kernel if you want mess with kernel modules (or uprade the kernel out of MS update cycle), though that is not necessary for this.
4. Uses a Distro not officially supported by CUDA (Kali Linux).
5. Has limited resources if compared to the host OS (although performance-wise, it has advantages in many areas).

If any of the above is not true for you, you may have an easier time and/or may skip some of the specifics (for example, instead of installing WSL CUDA, use a Windows CUDA toolkit).
Hopefully the instructions here are more compreensive to be able to help edge-case lovers like me and that it will be enough for simpler or similar scenarios. Sometimes it can be a bit too frustrating.

For convenience the .wslconfig file used for these tests has been included in this repo (if you decide to use WSL2).

- CPU(s): 20 (Intel(R) Core(TM) i9-12900H CPU @ 2.90GHz) - 10 cores made available for WSL2
- GPU(s): NVIDIA GeForce RTX 3080 Laptop GPU (8 GB VRAM) + Intel(R) UHD Graphics (16 GB Shared Video RAM)
- RAM: 32 GB DODIMM 3200 MT/s (24 GB for WSL2 and 32 GB swap)
- Hard-drive: High-speed NVMe SSD
- Host: Windows 11 (10.0.22635.4010)
- Virtualization Environment: WSL2 (2.3.17.0 Kernel: 5.15.153.1-2)
- Distro: Kali GNU/Linux Rolling (2024.3) x86_64
- Python 3.12.4

## Quick-start

1. Ensure CUDA Toolkit ([WSL](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#wsl), [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)) is setup and that it is working.

2. Follow the [FLUX - local installation guide](https://github.com/black-forest-labs/flux?tab=readme-ov-file#local-installation) for flux.

3. Install the extra packages by running the following commands:

  ```shell
  cd $HOME/flux # or any other directory you chose in step 1
  pip install git+https://github.com/huggingface/diffusers.git
  pip install optimum-quanto
  pip install -U transformers accelerate peft
  ```

4. Run this script inside the flux directory:
  
  ```shell
  python flux_generator.py --model schnell "A dirty beggar wearing old rags, sitting on the sidewalk with its back touching a graffiti wall and a street dog laying by his right side. An RGB lit gaming laptop lays on his lap. To his left, laying against the wall, a sign that reads 'My laptop cannot run a transformer'. Next to the sign a shallow box with several tossed coins."
  ```

5. To see other options for this script:

  ```shell
  $python ./flux_generator.py --help
  usage: flux_generator.py [-h] [--model {schnell,dev}] [--device {cuda,cpu}] [--offload_off] [--steps STEPS] [--width WIDTH] [--height HEIGHT] [--output OUTPUT] [--cuda_pipeline] prompt

  Flux Prompt to Image Generator
  
  positional arguments:
    prompt                The prompt to generate the image
  
  options:
    -h, --help            show this help message and exit
    --model {schnell,dev}
                          Which Flux model to use
    --device {cuda,cpu}   Device to use for inference (default: cuda - better performance)
    --offload_off         Disables model offload to CPU and uses the GPU as needed (default: offload is enabled). Lowers VRAM usage.
    --steps STEPS         Number of inference steps. Higher, yields better image quality and is proportionally slower. Default 4 for schnell and 20 for dev
    --width WIDTH         Output image Width (default: 1024)
    --height HEIGHT       Output image Height (default: 1024)
    --output OUTPUT       Output file name (default: flux-<model_name>.png)
    --cuda_pipeline       Enables CUDA for loading the entire pipeline, should perform better, but uses more VRAM. (default: cpu - lower VRAM usage)
  ```
## Benchmark with default parameters

These are the benchmarks with what I think are __close__ to the "bare minimum" to run both models. Maybe if you have a lower end hardware the `dev` model may not work.
So it is best to start with `schnell` and if you succeed with that, try the `dev` model, which is more memory intensive.

### Model: schnell ≈ 10min (4 steps)

![flux-schnell_beggar_developer](https://github.com/user-attachments/assets/fe10e1e5-d911-4092-9a14-4e9842894d22)

### Model: dev ≈ 40min (20 steps)

![flux-dev_beggar_developer](https://github.com/user-attachments/assets/f1c19d8a-8c7b-4315-b556-fdeccd6f662b)

## Tips for WSL2 users

- By default WSL is given only 2 dedicated cores (even if you have 20). Similarly, a very limited amount of RAM and virtual memory (or in linux terms, swap memory). But you can tweak that. To do so you need to create `.wslconfig` file (not created by default) in `%USERPROFILE%` folder (press Ctrl+R and paste it there). The final path is `C:\Users\<youruser>\.wslconfig`. After that you will need to (in powershell) `wsl --shutdown` to apply the settings.
- Nvidia documentation is not very clear about the CUDA Toolkit installation process for WSL. There may be hiccups, especially in scenarios where the Linux distribution is not officially supported by Nvidia. So your mileage may vary when going through the referenced documentation.
- Important points to note:
  - Do not install the NVIDIA driver for Linux.
  - If using a Debian-based distribution, use the WSL-Ubuntu CUDA Toolkit installation: [CUDA Toolkit installation guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu).
  - If not using a Debian-based distribution, read the documentation carefully as the CUDA Toolkit may overwrite the drivers installed by Windows for WSL2.
  - The Windows driver installation also installs the WSL2 driver, so there is no need to worry about it.
  - Run the command `nvidia-smi` to check if the driver is installed correctly (not `nvidia-detect` as mentioned in the Nvidia docs).
  - GPU memory usage will show as N/A in WSL2 because the controller is in Windows. You can view the GPU memory usage in Task Manager or Resource Monitor:

![Screenshot 2024-08-18 091414](https://github.com/user-attachments/assets/a0c67aa1-edbe-4a1b-802c-bbdd9e3b70ae)

## TODO

Other improvements I have considered, but didn't get to (and don't know if I will):

- [Tiny Autoencoder](https://huggingface.co/docs/diffusers/main/en/optimization/fp16#tiny-autoencoder)
- Attempt more up-to-date tokenizers. I tried `openai/clip-vit-base-patch32` and `google/siglip-so400m-patch14-384` but they led to runtime errors, I did not want to go deeper for now.
