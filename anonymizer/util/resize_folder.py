import file_utils
import image_utils
import os
from PIL import Image
import math

INPUT_PATH = "/Users/viktorro/Documents/report/figures/results/cityscapes_finals"
#OUTPUT_PATH = "/Users/viktorro/Documents/report/figures/results/resized"
OUTPUT_PATH = f"{INPUT_PATH}/resized"

# Takes priority over MAX_WIDTH / MAX_HEIGHT
RESIZE_RATIO = 0.85
#RESIZE_RATIO = None

MAX_WIDTH = 2000
MAX_HEIGHT = 2000
QUALITY = 65 # Quality of JPG 0-100

FORCE_JPG = True

SKIP_ALREADY_PROCESSED = False

sorted_filenames = sorted(os.listdir(INPUT_PATH))
num_images = len(sorted_filenames)


def _img_exists_any_ext(fname):
    exts = ["png", "jpg", "jpeg"]
    fname = file_utils.remove_file_ext(fname)

    for e in exts:
        if os.path.exists(f"{OUTPUT_PATH}/{fname}.{e}"):
            return True
    
    return False


def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

total_input_size = 0
total_output_size = 0

for index, fname in enumerate(sorted_filenames):
    if SKIP_ALREADY_PROCESSED and _img_exists_any_ext(fname):
        print(f"Skipping {fname} ({index}/{num_images})")
        continue

    input_path = f"{INPUT_PATH}/{fname}"

    if os.path.isdir(input_path):
        continue
    
    ext = fname.split(".")[-1]
    fname = "".join(fname.split(".")[:-1])

    img = Image.open(input_path)
    input_pixel_size = img.size

    input_size = os.path.getsize(input_path)
    total_input_size += input_size
    input_size_str = convert_size(input_size)

    if RESIZE_RATIO != None:
        img = img.resize((int(img.width * RESIZE_RATIO), int(img.height * RESIZE_RATIO)))
    else:    
        img = image_utils.resize_to_fit(img, MAX_WIDTH, MAX_HEIGHT)

    output_fname = f"{fname}"

    if (FORCE_JPG):
        ext = "jpg"
        print(f"{output_fname}.{ext}")
        file_utils.save_img(img, output_fname, OUTPUT_PATH, file_ext=ext, quality=QUALITY)
    else:
        file_utils.save_img(img, output_fname, OUTPUT_PATH)

    output_size = os.path.getsize(f"{OUTPUT_PATH}/{output_fname}.{ext}")
    total_output_size += output_size
    output_size_str = convert_size(output_size)

    print(f"Resized {fname} from {input_size_str} ({input_pixel_size}) to {output_size_str} ({img.size}) ({index+1}/{num_images})\n")
    
print(f"\nResized all files in folder, from {convert_size(total_input_size)} to {convert_size(total_output_size)} ({round((total_output_size / total_input_size)*100, 0)}% of input size)")