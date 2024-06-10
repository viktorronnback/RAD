import file_utils
import image_utils
import os
from PIL import Image

INPUT_PATH = "input/privacy-eval-uhd"
OUTPUT_PATH = "input/privacy-eval-tmp"

MAX_WIDTH = 1920
MAX_HEIGHT = 1920

SKIP_ALREADY_PROCESSED = True

sorted_filenames = sorted(os.listdir(INPUT_PATH))
num_images = len(sorted_filenames)

def _img_exists_any_ext(fname):
    exts = ["png", "jpg", "jpeg"]
    fname = file_utils.remove_file_ext(fname)

    for e in exts:
        if os.path.exists(f"{OUTPUT_PATH}/{fname}.{e}"):
            return True
    
    return False


for index, fname in enumerate(sorted_filenames):
    if SKIP_ALREADY_PROCESSED and _img_exists_any_ext(fname):
        print(f"Skipping {fname} ({index}/{num_images})")
        continue

    img = Image.open(f"{INPUT_PATH}/{fname}")

    if (img.width < 512 or img.height < 512):
        print(f"WARNING! Stable diffusion handles low resolution ({img.width} x {img.height} < 512 x 512) images poorly, consider using a super resolution tool such as Real-ESRGAN")
    
    img = image_utils.preprocess_image(img, MAX_WIDTH, MAX_HEIGHT)

    print(f"Preprocessed {fname} ({index+1}/{num_images})\n")

    file_utils.save_img(img, fname, OUTPUT_PATH, "png")
    