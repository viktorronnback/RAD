"""
Script for conversion between cityscapes annotation format and yolo-format.
"""

import json
import os, shutil
import numpy as np

FOLDERS = ["val", "train", "test"]

GT_FOLDER = "../datasets/cityscapes_unfiltered/{FOLDER}/"
SAVE_FOLDER = "../datasets/cityscapes_unfiltered_groups_as_single/labels/{FOLDER}/"

# Cityscape classes to COCO classes translations where possible. 
# out of roi - out of "region of interest"
conversions = {"rider": "person", "person": "person", "road": None, "sidewalk": None, "parking": None, "rail track": None, 
               "car": "car", "truck": "truck", "bus": "bus", "train": "train", "on rails": "train", "motorcycle": "motorcycle", "bicycle": "bicycle", 
               "caravan": "car", "trailer": "car", "building": None, "wall": None, "fence": None, "guard rail": None, "bridge": None, 
               "tunnel": None, "pole": None, "polegroup": None, "traffic sign": None, "traffic light": "traffic light", "vegetation": None, 
               "terrain": None, "sky": None, "ground": None, "dynamic": None, "static": None, "license plate": None, "ego vehicle": None,
               "out of roi": None, "rectification border": None, "persongroup": None, "cargroup": None, "bicyclegroup": None,
               "ridergroup": None, "motorcyclegroup": None, "truckgroup": None}

class_labels = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 4, "train": 5, "truck": 6, "traffic light": 7}

def confirm_y_n(msg: str) -> bool:
    print(f"{msg} [Y/n]")
    answer = input().lower()

    if answer == "y":
        return True
    
    return False

for folder in FOLDERS:
    gt_folder = GT_FOLDER.replace("{FOLDER}", folder)
    save_folder = SAVE_FOLDER.replace("{FOLDER}", folder)

    print(f"Converting files in {gt_folder}")

    if os.path.exists(save_folder):
        # Folder exists
        resp = confirm_y_n(f"There already exists a save folder {save_folder}. Do you want to overwrite it?")

        if resp:
            shutil.rmtree(save_folder)
        else:
            exit(0)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    files = os.listdir(gt_folder)

    assert len(files) != 0, f"No files to convert in {gt_folder}"

    files_processed = 0

    for filename in files:
        if filename == ".DS_Store" or not "json" in filename:
            continue

        files_processed += 1

        f = open(gt_folder + filename)
        data = json.load(f)
        
        output_filename = save_folder + filename.replace("gtFine_polygons.json", "leftImg8bit.txt")

        with open(output_filename, "w") as file:
            height = data["imgHeight"]
            width = data["imgWidth"]

            for object in data["objects"]:
                label: str = object["label"]

                # Treat groups as single instances.
                if "group" in object["label"]:
                    label = label.replace("group", "")

                # Disregard labels which are ignored. 
                if conversions[label] is None:
                    continue
                
                label = conversions[label] # String
                class_label = class_labels[label] # Integer

                file.write(str(class_label) + " ")

                for x, y in object["polygon"]:
                    x_norm = np.round(x/width, 3)
                    y_norm = np.round(y/height, 3)
                    file.write(str(x_norm) + " " + str(y_norm) + " ")

                file.write("\n")
        
        print(f"Converted {output_filename}")
    
    print(f"Converted {files_processed} files")
    print("\n")

