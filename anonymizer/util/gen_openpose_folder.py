import controlnet_utils # Import does not work...
import os
from PIL import Image

INPUT_PATH = "input/privacy-eval"

sorted_filenames = sorted(os.listdir(INPUT_PATH))
num_images = len(sorted_filenames)


for index, fname in enumerate(sorted_filenames):
    img = Image.open(f"{INPUT_PATH}/{fname}")

    controlnet_utils.gen_pose(img)

    print(f"Processed {fname} ({index} / {num_images})")