import yaml
import torch
from os import path

import util.file_utils as f_util
from util.error_print import error

DEFAULT_PARENT_INPUT_FOLDER = "./input"
DEFAULT_PARENT_OUTPUT_FOLDER = "./output"


class Settings:
    def __init__(self, yaml_path="config.yaml"):
        self.yaml_path = yaml_path

        # Define settings below, 'key: value' represent 'setting: default value'
        # Use None for required settings (no default)
        accepted_settings = {
            'input': None,
            'output': None,
            'overwrite': False,
            'save_format': 'default',
            'strength': None,
            'controlnet_conditioning_scale': None,
            'guidance_scale': None,
            'inference_steps': None,
            'prompt': '',
            'negative_prompt': '',
            'seed': -1,
            'controlnet': None,
            'canny_min_threshold': 175,
            'canny_max_threshold': 350,
            'canny_silhouette': False,
            'diffusion_model': "SG161222/RealVisXL_V3.0",
            'refiner_model': '',
            'controlnet_canny_model': "diffusers/controlnet-canny-sdxl-1.0",
            'controlnet_pose_model': "thibaud/controlnet-openpose-sdxl-1.0",
            'yolo_obj_det_model': "models/yolov8x.pt",
            'sam_seg_model': "models/sam_vit_h_4b8939.pth",
            'max_output_width': 1920,
            'max_output_height': 1080,
            'cpu_offloading': 0,
            'compile_unet': True,
            'verbose': False,
            'measure_time': False,
            'device': self._default_device(),
            'segmentation': "default"
        }

        # Set default settings
        for key, value in accepted_settings.items():
            setattr(self, key, value)
        
        self._load_yaml_settings(accepted_settings)

        self._post_process()

        self._validate_settings()

    def _load_yaml_settings(self, accepted_settings: dict):
        """ Sets default settings as attributes, overwrites these with YAML """
        # Load YAML file
        try:
            with open(self.yaml_path, 'r') as file:
                yaml_settings = yaml.safe_load(file) or {}
        except FileNotFoundError:
            error(f"Settings file not found at {self.yaml_path}")
        
        # Make sure there are no erronious settings
        for key in yaml_settings:
            if key not in accepted_settings:
                key_string = ", ".join(accepted_settings.keys())
                error(f"Invalid setting '{key}'\n\nAccepted settings: {key_string}")

        # Apply settings from yaml (overwrite default)
        for key, value in accepted_settings.items():
            if key in yaml_settings:
                setattr(self, key, yaml_settings[key])
            elif value == None:
                error(f"Missing required setting '{key}'")
            

    def _set_seg_settings(self):
        self.seg_device: str = self.device
        
        if self.cpu_offloading == 3:
            # If using maximum offloading, perform segmentation on CPU
            # Will be overwritten by specific setting if not using default/seq
            self.seg_device = "cpu"

        match self.segmentation:
            case "default":
                self.seg_parallel: bool = True
            case "seq":
                self.seg_parallel: bool = False
            case _:
                self.seg_parallel: bool = True
                self.seg_device = self.segmentation
        
        print(f"Segmentation on device {self.seg_device}, in parallel {self.seg_parallel}")

    def _post_process(self):
        """ Post-process settings """
        # Add default parent folders to input / output paths (if not absolute)
        self.input = self._parent_io_path(self.input, DEFAULT_PARENT_INPUT_FOLDER)
        self.output = self._parent_io_path(self.output, DEFAULT_PARENT_OUTPUT_FOLDER)

        # Segmentation
        self._set_seg_settings()

    def _default_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            # Mac m1/m2
            return "mps"

        return "cpu"

    def save_copy_to_output(self):
        """ Saves copy of yaml file to output folder """
        f_util.copy_file(self.yaml_path, self.output)

    def _parent_io_path(self, current_path: str, default_parent_folder: str) -> str:
        """ Returns updated path with parent folder or unchanged path if path is absolute """
        if not f_util.is_absolute(current_path):
            return path.join(default_parent_folder, current_path)

        return current_path

    def _validate_settings(self):
        if not f_util.path_exists(self.input):
            error("Invalid input path")

        if self.canny_min_threshold > self.canny_max_threshold:
            error("canny_min_threshold cannot be larger than canny_max_threshold")

        if self.compile_unet and self.cpu_offloading != 0:
            error("compile_unet can only be true if cpu_offloading is 0")

        if self.refiner_model and self.cpu_offloading != 0:
            error("refiner_model cannot be used with cpu_offloading")