# Basic Imports
import os
from datetime import datetime
import importlib.resources as pkg_resources

### SD Demo Imports
# Custom MRI Data Service
import app.mri_service.config as config
from app.api.models.mri_service import MRIProcessParameters

# ControlNet Imports
from controlnet.annotator.util import resize_image, HWC3
from controlnet.annotator.canny import CannyDetector
from controlnet.cldm.model import create_model, load_state_dict
from controlnet.cldm.ddim_hacked import DDIMSampler

# General Imports
import cv2
import einops
import numpy as np
import random
import torch

from pytorch_lightning import seed_everything
import imageio
import numpy as np
import matplotlib.pyplot as plt

# Initialize Canny Detector (Edge Detection)
apply_canny = CannyDetector()

# Load the model
model_yaml_path = pkg_resources.path("controlnet.models", "cldm_v15.yaml")
model = create_model(model_yaml_path).cpu()
model_path = os.path.join(os.path.dirname(__file__), "models", "control_sd15_canny.pth")
try:
    state_dict = torch.load(model_path, map_location="cpu")  # Load safely
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

# Determine device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights with proper device mapping
model.load_state_dict(load_state_dict(model_path, location=device))

# Move model to the selected device
model = model.to(device)

ddim_sampler = DDIMSampler(model)

### Load reference image
# Load the PNG image into a numpy array
image_path = os.path.join(os.path.dirname(__file__), "input_images", "mri_brain.jpg")
input_image = imageio.imread(image_path)

# image processing functions
def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
def lab2rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
def rgb2yuv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
def yuv2rgb(yuv: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
def srgb2lin(s):
    s = s.astype(float) / 255.0
    return np.where(
        s <= 0.0404482362771082, s / 12.92, np.power(((s + 0.055) / 1.055), 2.4)
    )
def lin2srgb(lin):
    return 255 * np.where(
        lin > 0.0031308, 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055, 12.92 * lin
    )
def get_luminance(
    linear_image: np.ndarray, luminance_conversion=[0.2126, 0.7152, 0.0722]
):
    return np.sum([[luminance_conversion]] * linear_image, axis=2)
def take_luminance_from_first_chroma_from_second(luminance, chroma, mode="lab", s=1):
    assert luminance.shape == chroma.shape, f"{luminance.shape=} != {chroma.shape=}"
    if mode == "lab":
        lab = rgb2lab(chroma)
        lab[:, :, 0] = rgb2lab(luminance)[:, :, 0]
        return lab2rgb(lab)
    if mode == "yuv":
        yuv = rgb2yuv(chroma)
        yuv[:, :, 0] = rgb2yuv(luminance)[:, :, 0]
        return yuv2rgb(yuv)
    if mode == "luminance":
        lluminance = srgb2lin(luminance)
        lchroma = srgb2lin(chroma)
        return lin2srgb(
            np.clip(
                lchroma
                * ((get_luminance(lluminance) / (get_luminance(lchroma))) ** s)[
                    :, :, np.newaxis
                ],
                0,
                1,
            )
        )
 
# Main inference function
def process(input_image, 
            prompt, a_prompt, n_prompt, 
            num_samples, image_resolution, ddim_steps, 
            guess_mode, strength, scale, seed, eta, 
            low_threshold, high_threshold):
    '''Run model for inference with provided parameters'''
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # Sample Model
        # TODO: Research more on the parameters
        model.control_scales = [
            strength * (0.825 ** float(12 - i)) for i in range(13)] \
                if guess_mode else ([strength] * 13)  
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

def generate_synthetic_mri_images(params: MRIProcessParameters) -> dict:
    """
    Generate MRI Images.
    """
    result = process(
        input_image=input_image,
        prompt=params.prompt,
        a_prompt=params.a_prompt,
        n_prompt=params.n_prompt,
        num_samples=params.num_samples,
        image_resolution=params.image_resolution,
        ddim_steps=params.ddim_steps,
        guess_mode=params.guess_mode,
        strength=params.strength,
        scale=params.scale,
        seed=params.seed,
        eta=params.eta,
        low_threshold=params.low_threshold,
        high_threshold=params.high_threshold
    )

    for res in result:
        plt.imshow(res)
        plt.axis(False)
        plt.show()
        
    index = -1 # only show the last result
    test = take_luminance_from_first_chroma_from_second(
        resize_image(HWC3(input_image), params.image_resolution), 
        result[index], mode="lab")

    # Plot result
    fig, axs = plt.subplots(1,3, figsize=(15, 5))
    axs[0].imshow(input_image)
    axs[1].imshow(result[index])
    axs[2].imshow(test)

    axs[0].axis(False)
    axs[1].axis(False)
    axs[2].axis(False)

    # Create a unique directory for the results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.path.dirname(__file__), "output", f"result_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    comparison_path = os.path.join(output_dir, "comparison.png")
    fig.savefig(comparison_path)
    plt.close(fig)

    # Save individual results
    # result_paths = []
    # for i, res in enumerate(result):
    #     result_path = os.path.join(output_dir, f"result_{i}.png")
    #     imageio.imwrite(result_path, res)
    #     result_paths.append(result_path)

    result_paths = os.path.join(output_dir, f"result.png")
    imageio.imwrite(result_paths, result[-1])

    return {
        "output_dir": output_dir,
        "comparison_image": comparison_path,
        "result_images": result_paths
    }
