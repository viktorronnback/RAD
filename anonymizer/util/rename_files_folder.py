import os

FOLDER_PATH = "input/privacy-eval"

sorted_filenames = sorted(os.listdir(FOLDER_PATH))
num_files = len(sorted_filenames)

for index, fname in enumerate(sorted_filenames):
    ext = fname.split(".")[-1]
    new_name = fname.replace(ext, "")

    # How filename is changed
    new_name = new_name.replace(" ", "_")
    new_name = new_name.replace("__", "_")
    new_name = new_name.lower()
    new_name = new_name.replace(".", "")

    new_name = f"{new_name}.{ext}"

    if fname == new_name:
        continue

    os.rename(f"{FOLDER_PATH}/{fname}", f"{FOLDER_PATH}/{new_name}")

    print(f"Renamed {fname} to {new_name} in {FOLDER_PATH} ({index+1}/{num_files})")