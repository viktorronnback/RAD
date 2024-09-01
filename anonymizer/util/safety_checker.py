from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
import numpy as np
import torch
from PIL import Image

from util import settings_handler

torch_dtype = torch.float16

class SafetyChecker:
    """ Modified from https://discuss.huggingface.co/t/sdxl-safety-checker/49633/3 """
    
    def __init__(self, settings: settings_handler.Settings):
        self.device = settings.device
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to(self.device)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    def is_safe(self, image: Image.Image) -> bool:
        images = [image] # Convert to list of images
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
        images_np = [np.array(img) for img in images]

        _, has_nsfw_concepts = self.safety_checker(
            images=images_np,
            clip_input=safety_checker_input.pixel_values.to(self.device),
        )

        return has_nsfw_concepts[0] == False