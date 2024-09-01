from PIL import Image
import numpy as np
import cv2
from util.controlnet_utils import gen_canny

def erode_mask(image_mask: Image.Image, kernel_size: int=3):
    image = np.array(image_mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(image, kernel)
    eroded = Image.fromarray(eroded)
    return eroded

def dilate_mask(image_mask: Image.Image, kernel_size: int=3):
    image = np.array(image_mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(image, kernel)
    dilated = Image.fromarray(dilated)
    return dilated

def get_concat_v(im1, im2):
    """ Concatenates two images vertically. Source: https://note.nkmk.me/en/python-pillow-concat-images/ """
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def stitch_images(foreground_image: Image.Image, background_image: Image.Image, mask_image: Image.Image):
    """ Stitches foreground image onto background image using the mask image. 

    Returns:
        background (Image.Image): The stitched image.   
    """
    assert foreground_image.size == background_image.size == mask_image.size, f"Image sizes must match! {foreground_image.size} == {background_image.size} == {mask_image.size}"

    # Save edge image for use in blending.
    canny = gen_canny(mask_image, 100, 300)
    dilated_canny = dilate_mask(canny, 5)

    # Convert images to RGBA.
    foreground = foreground_image.convert("RGBA")
    background = background_image.convert("RGBA")

    # Get data arrays.
    front_data = foreground.getdata()
    background_data = background.getdata()
    mask_data = mask_image.getdata()
    canny_data = dilated_canny.getdata()

    new_data = []

    # Combine foreground and background data. 
    for idx, item in enumerate(mask_data): 
        if item == 0:  # Black mask pixel
            new_data.append(background_data[idx])
        else:
            new_data.append(front_data[idx])

    # Update background image with new data. 
    background.putdata(new_data)

    # Make a blurry version. 
    blurry_array = cv2.GaussianBlur(np.asarray(background), ksize=(5, 5), sigmaX=0)
    blurry_image = Image.fromarray(blurry_array)
    blurry_data = blurry_image.getdata()

    combined_data = []

    # Combine blurry image in edges with new data.
    for idx, item in enumerate(canny_data):
        if item == (0, 0, 0):
            combined_data.append(new_data[idx])
        else:
            combined_data.append(blurry_data[idx])

    background.putdata(combined_data)
    return background