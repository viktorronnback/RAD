from PIL import Image
import cv2
import numpy as np

PARENT_OUTPUT_FOLDER = "./output"


def resize_ratio(image: Image.Image, ratio: float, resample=Image.Resampling.BICUBIC) -> Image.Image:
    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)

    return image.resize(size=(new_width, new_height), resample=resample)


def crop(image: Image.Image, x: int, y: int, size: int):
    image = image.crop((x, y, x+size, y+size))
    return image


def dilate_mask(image_mask: Image.Image, size: int):
    image = np.array(image_mask)
    kernel = np.ones((size, size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=2)
    dilated = Image.fromarray(dilated)

    return dilated


def crop_to_8(image: Image.Image) -> Image.Image:
    """ Crops image to have width/height be divisible by 8 """
    w_remainder = image.width % 8
    h_remainder = image.height % 8

    w_cropped = image.width - w_remainder
    h_cropped = image.height - h_remainder

    if (w_remainder != 0 and h_remainder != 0):
        print(f"WARNING! Image w/h is not divisible by 8, cropping right and/or bottom by (-{w_remainder}, -{h_remainder}) px")

    return image.crop((0, 0, w_cropped, h_cropped))


def resize_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    original_width, original_height = image.size
    ratio = min(max_width / original_width, max_height / original_height)
    
    # If ratio < 1, the image needs to be resized
    if ratio < 1:
        resized_image = resize_ratio(image, ratio)
        return resized_image
    else:
        # If the image is already within the maximum dimensions, return it unchanged
        return image


def preprocess_image(image: Image.Image, max_width: int = 0, max_height: int = 0) -> Image.Image:
    """ Performs necessary preprocessing of image to work in pipeline """
    resized = image
    
    if max_width != 0 and max_height != 0:
        # Resize image to not exceed GPU memory
        resized = resize_to_fit(image, max_width, max_height)
    
    # Crop so w/h is divisible by 8 due to generator requirement
    cropped = crop_to_8(resized)

    return cropped


def create_collage(images: list[Image.Image]) -> Image.Image:
    """ Saves images to 2x2 collage. """
    widths, heights = zip(*(i.size for i in images))

    total_width = 2*widths[0]
    total_height = 2*heights[0]

    new_im = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    y_offset = 0

    for im in images:
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
        if x_offset == 2*im.size[0]:
            x_offset = 0
            y_offset += im.size[1]

    return new_im
