""" Hybrid diffusion model in FHE.

Original file is located at
https://colab.research.google.com/drive/1DFq9AI85hPOmHs2EFR0-gX2YdNpTA5z9
"""

import random

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from torch.cuda import seed_all
from tqdm.auto import tqdm

from concrete.ml.torch.hybrid_model import HybridFHEModel


# We need to wrap the model to set defaults parameters
# as to have only one encrypted parameter
class Wrapper(torch.nn.Module):
    def __init__(self, model, timestep, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodule = model
        # implement fqallback
        self.config = model.config
        self.timestep = timestep

    def forward(self, inputs, timestep, **kwargs):
        return self.submodule.forward(inputs, timestep=timestep, **kwargs)


def seed_everything(seed):
    random.seed(seed)
    seed += 1
    np.random.seed(seed % 2**32)
    seed += 1
    torch.manual_seed(seed)
    seed += 1
    torch.use_deterministic_algorithms(True)
    return seed


def generate_image(
    model,
    scheduler,
    output: str = "generated.png",
    timestep: int = 1,
    seed=None,
    fhe=None,
    device=None,
):
    if seed is not None:
        seed_everything(seed)

    sample_size = model.config.sample_size
    scheduler.set_timesteps(timestep)
    noise = torch.randn((1, 3, sample_size, sample_size), device=device)
    input = noise

    for t in tqdm(scheduler.timesteps, total=timestep):
        with torch.no_grad():
            kwargs = {}
            if fhe is not None:
                kwargs = {"fhe": fhe}
            noisy_residual = model(input, t, **kwargs).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input: torch.Tensor = prev_noisy_sample

    image = (input / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))

    with open(output, "wb") as file:
        image.save(file)

    return image


if __name__ == "__main__":
    # TODO: handle multi-device execution in Hybrid Model
    # For now we only rely on the CPU
    # todo: handle other devices in HybridFHEModel
    # model = model.to(torch.device("cpu")

    device_str = "cpu"  # default
    if torch.backends.mps.is_available():
        device_str = "mps"
    if torch.cuda.is_available():
        device_str = "cuda"
    # device_str = "cpu"
    device = torch.device(device_str)

    # Create objects
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
    model = UNet2DModel.from_pretrained("google/ddpm-cat-256", device=device)
    model.eval()
    model = model.to(device)
    sample_size = model.config.sample_size
    timestep = 100

    wrapped_model = Wrapper(model, timestep=timestep)
    wrapped_model.to(device)
    submodule_names = [
        layer_name
        for (layer_name, layer) in wrapped_model.named_modules()
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d))
    ]
    print(f"{len(submodule_names)}")

    # Random selection of a submodule -> we could try with all submodules
    index = 25
    print(submodule_names[index])
    fhe_submodules = [submodule_names[index]]

    hybrid_model = HybridFHEModel(
        wrapped_model,
        fhe_submodules,
        verbose=2,
    )

    # Create a hybrid model
    compile_size = 3
    inputs = torch.randn((compile_size, 3, sample_size, sample_size), device=device)
    timesteps = torch.arange(0, compile_size, device=device)
    print("compiling model")
    hybrid_model.compile_model(
        inputs,
        timesteps,
        n_bits=8,
    )

    print(hybrid_model)

    # Generate torch image as reference
    generate_image(
        hybrid_model,
        scheduler,
        output="hybrid.png",
        timestep=timestep,
        fhe="simulate",
        device=device,
    )
    generate_image(
        model,
        scheduler,
        output="debug.png",
        timestep=timestep,
        device=device,
    )
    generate_image(
        wrapped_model,
        scheduler,
        timestep=timestep,
        output="torch.png",
        device=device,
    )
