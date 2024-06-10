from PIL import Image
import os

import stitch


def stitch_folder(input_folder: str):
    """ Stitches images in /generated onto /imgs using /annotations. """
    output_folder = input_folder + "/stitched/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    imgs_folder = input_folder + "/imgs"
    annotations_folder = input_folder + "/annotations"
    generated_folder = input_folder + "/generated"

    # Sort files in order and filter out hidden files.
    generated_list = sorted((f for f in os.listdir(
        generated_folder) if not f.startswith(".")), key=str.lower)
    original_list = sorted((f for f in os.listdir(
        imgs_folder) if not f.startswith(".")), key=str.lower)
    annotations_list = sorted((f for f in os.listdir(
        annotations_folder) if not f.startswith(".")), key=str.lower)

    for front_filename, back_filename, mask_filename in zip(generated_list, original_list, annotations_list):
        front_image = Image.open(generated_folder + "/" + front_filename)
        background_image = Image.open(imgs_folder + "/" + back_filename)
        mask_image = Image.open(annotations_folder + "/" + mask_filename)

        front_image = front_image.resize((1920, 1080))
        background_image = background_image.resize((1920, 1080))
        mask_image = mask_image.resize((1920, 1080))

        final_image = stitch.stitch_images(
            front_image, background_image, mask_image)
        final_image.save(output_folder + back_filename, format="png")


if __name__ == "__main__":
    stitch_folder("../input/lindy_hop/samples")
