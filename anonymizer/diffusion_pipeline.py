from PIL import Image
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
from util import settings_handler
from util import time_module
from util.error_print import error


class DiffusionPipeline:
    def __init__(self, settings: settings_handler.Settings):
        self.settings: settings_handler.Settings = settings

        self.generator = torch.Generator(device=settings.device)

        if settings.seed != -1:
            print(f"Using manual seed {settings.seed}")
            # Same seed, same prompt (same model) yields identical result (deterministic)
            self.generator = self.generator.manual_seed(settings.seed)

        # Control net
        if settings.controlnet == "canny":
            print("Using canny controlnet model")
            controlnet = ControlNetModel.from_pretrained(settings.controlnet_canny_model, variant="fp16", torch_dtype=torch.float16) # Canny
        elif settings.controlnet == "pose":
            print("Using openpose controlnet model")
            controlnet = ControlNetModel.from_pretrained(settings.controlnet_pose_model, torch_dtype=torch.float16)  # OpenPose
        elif settings.controlnet == "both":
            controlnet = [ControlNetModel.from_pretrained(settings.controlnet_canny_model, variant="fp16", torch_dtype=torch.float16)] # Canny
            controlnet.append(ControlNetModel.from_pretrained(settings.controlnet_pose_model, torch_dtype=torch.float16))  # OpenPose
        else:
            error(f"Invalid controlnet setting '{settings.controlnet}'")

        # VAE
        # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        # Load pipeline
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            settings.diffusion_model,
            controlnet=controlnet,
            variant="fp16",
            torch_dtype=torch.float16
        )
        
        if settings.refiner_model:
            print("Using refiner")
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                settings.refiner_model,
                text_encoder_2=pipe.text_encoder_2,
                vae=pipe.vae,
                use_safetensors=True,
                variant="fp16",
                torch_dtype=torch.float16
            )

            refiner = refiner.to(settings.device)

            if settings.compile_unet:
                print("Compiling refiner Unet")
                refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

            self.refiner: StableDiffusionXLImg2ImgPipeline = refiner
        
        # Scheduler
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("\nPipeline scheduler:\n", pipe.scheduler)

        # VRAM reduction (at the cost of speed)
        if settings.cpu_offloading == 1:
            pipe.enable_model_cpu_offload()
        elif settings.cpu_offloading == 2:
            pipe.enable_sequential_cpu_offload()
        elif settings.cpu_offloading == 3:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_tiling()
        else:
            pipe = pipe.to(settings.device)  # Only run if CPU loading is turned off
        
        print(f"Using offloading {settings.cpu_offloading}")

        if settings.compile_unet:
            print("Compiling Unet")
            t0 = time_module.start_time()
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            time_module.print_elapsed_seconds(settings, t0, "Compile Unet")

        self.pipe: StableDiffusionXLControlNetImg2ImgPipeline = pipe # Set pipe member variable


    def generate(self, input_image: Image.Image, input_controlnet: Image.Image | list[Image.Image]):
        """ Anonymizes a single frame """

        if type(input_controlnet) == list:
            assert input_image.size == input_controlnet[0].size == input_controlnet[1].size, f"Image sizes must match! {input_image.size} != {input_controlnet[0].size} != {input_controlnet[1].size}"
        else:
            assert input_image.size == input_controlnet.size, f"Image sizes must match! {input_image.size} != {input_controlnet.size}"

        common_params = {
            "prompt": self.settings.prompt,
            "negative_prompt": self.settings.negative_prompt,
            "num_images_per_prompt": 1,
            "original_size": input_image.size,
            "target_size": input_image.size,
            "width": input_image.width,
            "height": input_image.height,
            "num_inference_steps": self.settings.inference_steps,
            "strength": self.settings.strength,
            "guidance_scale": self.settings.guidance_scale,
            "generator": self.generator
        }

        controlnet_params = {
            "control_image": input_controlnet,
            "controlnet_conditioning_scale": self.settings.controlnet_conditioning_scale,
        }
        
        # Generate
        if self.settings.refiner_model:
            generated_image = self.pipe(
                image=input_image,
                denoising_end=0.8,
                output_type="latent",
                **common_params,
                **controlnet_params
            ).images[0]
            
            refined_image = self.refiner(
                image=generated_image,
                denoising_start=0.8,
                **common_params
            ).images[0]
            
            return refined_image
        else:
            generated_image = self.pipe(
                image=input_image,
                **common_params,
                **controlnet_params
            ).images[0]

            return generated_image
