import os
from PIL import Image


def create_cutout(img_file: str, filename: str, annotation_file: str, output_folder: str):
    """ Creates cutout in image from annotated segment and saves as png.  
    Arguments:
        img_file: path to img file.
        filename: name of file.
        annotation_file: path to annotation file.
        output_folder: folder where the cutout should be saved. 
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create transparent image only including segmented part.
    crop_img = Image.open(annotation_file)
    orig_img = Image.open(img_file)

    crop_rgba = crop_img.convert("RGBA")
    orig_rgba = orig_img.convert("RGBA")

    crop_datas = crop_rgba.getdata()
    orig_datas = orig_rgba.getdata()

    # New pixel data
    new_data = []

    # Look through each pixel in cropped data.
    for idx, item in enumerate(crop_datas):
        # finding black colour by its RGB value
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            # storing a transparent value when we find a black colour
            new_data.append((255, 255, 255, 0))
        else:
            # add original pixel value
            new_data.append(orig_datas[idx])

    crop_rgba.putdata(new_data)
    crop_rgba.save(
        f"{output_folder}/{filename.split('.')[0]}.png", "PNG")


def create_cutouts(input_folder: str, individual_masks: bool=False):
    """ Creates cutouts from segmented images. 
    Arguments:
        input_folder - folder with subfolders /imgs and /annotations to be used for cutouts. 
        individual_masks - whether or not to create cutouts for individual segmentations, 
        assumes that each img in /imgs has its own folder with individual masks. 
    """
    img_dir = input_folder + "/imgs"
    annotation_dir = input_folder + "/annotations"
    output_folder = input_folder + "/cutouts"

    img_list = sorted(filter(lambda x: x.endswith(".jpg"), os.listdir(img_dir)))
    annotation_list = sorted(filter(lambda x: x.endswith(".jpg"), os.listdir(annotation_dir)))

    for img_filename, annotation_filename in zip(img_list, annotation_list):
        img_file = os.path.join(img_dir, img_filename)
        annotation_file = os.path.join(annotation_dir, annotation_filename)

        # Make sure that these are files.
        if not os.path.isfile(img_file) or not os.path.isfile(annotation_file):
            raise Exception("not a file")

        create_cutout(img_file, img_filename, annotation_file, output_folder)

        if individual_masks:
            masks_folder = annotation_file.rstrip(".jpg")

            if not os.path.exists(masks_folder):
                raise Exception("Individual masks has not been created.")

            for mask_filename in os.listdir(masks_folder):
                mask_file = os.path.join(masks_folder, mask_filename) 
                create_cutout(img_file, mask_filename, mask_file, output_folder + "/" + img_filename.rstrip(".jpg"))


if __name__ == "__main__":
    # TODO argparse input folder
    # create_cutouts("../../Desktop/celebs", individual_masks=True)
    create_cutouts("../input/schoolgirls", individual_masks=False)
