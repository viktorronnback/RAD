from controlnet_aux import OpenposeDetector
from PIL import Image
import cv2
import numpy as np
import time
from util import file_utils

def gen_pose(input_image: Image.Image) -> Image.Image:
    salt = "pose"

    if file_utils.is_cached(input_image, salt=salt):
        print("Pose already generated")
        return file_utils.load_img_cache(input_image, salt=salt)

    start = time.time()

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    output_image = openpose(input_image=input_image, include_hand=True, include_face=True, image_resolution=input_image.height, detect_resolution=input_image.height)

    # Make sure output image is same size as input
    output_image = output_image.resize(input_image.size)

    # Save to cache
    file_utils.save_img_cache(original_img=input_image, cache_img=output_image, salt=salt)

    duration = time.time() - start

    print(f"Generated pose in {round(duration, 1)} second(s)\n")
    
    return output_image


def gen_canny(image: Image.Image, threshold1: float, threshold2: float) -> Image.Image:
    """ Generates a canny edge from image https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html """
    # get canny image
    image = np.array(image)
    image = cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)
    
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    
    canny_image = Image.fromarray(image)
    
    print("Generated canny\n")
    
    return canny_image
