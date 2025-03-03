from pydantic import BaseModel
from typing import Optional

class MRIProcessParameters(BaseModel):
    prompt: str = "mri brain scan"
    a_prompt: str = "good quality"
    n_prompt: str = "animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    num_samples: int = 1
    image_resolution: int = 512
    ddim_steps: int = 10
    guess_mode: bool = False
    strength: float = 1.0
    scale: float = 9.0
    seed: int = 1
    eta: float = 0.0
    low_threshold: int = 50
    high_threshold: int = 100
