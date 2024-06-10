import os, shutil

FOLDERS = ["val", "train", "test"]

ANNOT_FOLDER = "../datasets/cityscapes_filtered_no_groups/labels/{FOLDER}/"

GT_FOLDER = "../datasets/cityscapes_unfiltered/images/{FOLDER}/"
SAVE_FOLDER = "../datasets/cityscapes_filtered_no_groups/images/{FOLDER}/"

for folder in FOLDERS:
    annot_folder = ANNOT_FOLDER.replace("{FOLDER}", folder)
    gt_folder = GT_FOLDER.replace("{FOLDER}", folder)
    save_folder = SAVE_FOLDER.replace("{FOLDER}", folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print(f"Filtering out images in {gt_folder} using {annot_folder}")

    files = os.listdir(annot_folder)

    for fname in files:
        fname = fname.split(".")[0]
        
        copyfrom = f"{gt_folder}{fname}.png"
        copyto = f"{save_folder}{fname}.png"

        shutil.copyfile(copyfrom, copyto)

        print(f"Copied {fname} from {copyfrom} to {copyto}")
