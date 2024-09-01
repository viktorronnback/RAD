import os
import shutil
import json
import numpy as np

# IMG_FOLDER = "../datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/"
# GT_FOLDER = "../datasets/cityscapes/gtFine_trainvaltest/gtFine/val/"

# SAVE_FOLDER = "../datasets/cityscapes_filtered/val/"

IMG_FOLDER = "../datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/"
GT_FOLDER = "../datasets/cityscapes/gtFine_trainvaltest/gtFine/train/"

SAVE_FOLDER = "../datasets/cityscapes_unfiltered/images/train/"

def main():
    
    total_images = 0 
    images_saved = 0
    # For every image file and annotation file. 
    #   If annotation file contains a "person" object:
    #       Save all files to new IMG and ANNOTATION folders. 
    for folder_name in os.listdir(IMG_FOLDER):
        if folder_name == ".DS_Store":
            continue

        img_folder = IMG_FOLDER + folder_name + "/"
        annotation_folder = GT_FOLDER + folder_name + "/"

        for image_filename in os.listdir(img_folder):
            if image_filename == ".DS_Store":
                continue

            json_file = annotation_folder + image_filename.replace("leftImg8bit.png", "gtFine_polygons.json")

            f = open(json_file)
            data = json.load(f)

            save = True

            # # Save images containing a person object larger than 50px in height. 
            # for object in data["objects"]:
            #     if object["label"] == "person":
            #         y_values = [point[1] for point in object["polygon"]]
            #         y_min = np.min(y_values)
            #         y_max = np.max(y_values)

            #         if y_max - y_min > 50:
            #             save = True
            #         break

            if save:
                images_saved += 1

                save_annotation_folder = SAVE_FOLDER + "annotations/"
                save_img_folder = SAVE_FOLDER + "imgs/"

                if not os.path.exists(save_annotation_folder):
                    os.makedirs(save_annotation_folder)

                if not os.path.exists(save_img_folder):
                    os.makedirs(save_img_folder)

                shutil.copy(img_folder + image_filename, save_img_folder + image_filename)
                
                for annotation_file in [image_filename.replace("leftImg8bit", "gtFine_color"), 
                                        image_filename.replace("leftImg8bit", "gtFine_instanceIds"), 
                                        image_filename.replace("leftImg8bit", "gtFine_labelIds"),
                                        image_filename.replace("leftImg8bit.png", "gtFine_polygons.json")]:
                    shutil.copy(annotation_folder + annotation_file, save_annotation_folder + annotation_file)
            
            # Samples image and annotations naming:
            # aachen_000000_000019_leftImg8bit
            # aachen_000000_000019_gtFine_color.png
            # aachen_000000_000019_gtFine_instanceIds
            # aachen_000000_000019_gtFine_labelIds
            # aachen_000000_000019_gtFine_polygons
                    
            total_images += 1
    
    print("Total images in dataset:", total_images)
    print("Images containing people:", images_saved)



if __name__ == "__main__":
    main()