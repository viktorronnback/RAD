import os
import shutil
from PIL import Image
from hashlib import sha256

CACHE_FOLDER_ENV = 'ANONYMIZER_CACHE'
cache_folder = ".cache/" if CACHE_FOLDER_ENV not in os.environ else os.environ[CACHE_FOLDER_ENV]

if cache_folder != ".cache/":
    print(f"Using custom cache {cache_folder}")

CACHE_FILE_EXT = "png"

# TODO Replace path concatenation with os operations (to support Windows backslash paths)

def path_exists(path: str):
    return os.path.exists(path)


def create_path_if_not_exists(folder_path):
    """ Creates output folder if it does not already exist """
    if not path_exists(folder_path):
        os.makedirs(folder_path)


def remove_file_ext(path: str) -> str:
    filename = path.split("/")[-1]
    return filename.split(".")[0]


def _rgba_to_rgb(img: Image.Image) -> Image.Image:
    """ Converts transparency to white background """
    background = Image.new(mode="RGBA", size=img.size, color=(255, 255, 255))
    
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def _save_img_helper(img: Image.Image, filename: str, folder: str, file_ext: str = "png", quality: float = 80):
    save_as_jpg: bool = file_ext == "jpg" or file_ext == "jpeg"

    if file_ext != None:
        assert file_ext.find(".") == -1, "file_ext should not contain dot (.)"

        if save_as_jpg:
            img = _rgba_to_rgb(img)
            
    create_path_if_not_exists(folder)

    path = f"{folder}/{filename}.{file_ext}"

    print(f"Saving to {path}")

    if (save_as_jpg):
        img.save(path, format="JPEG", quality=quality)
    else:
        img.save(path)


def save_img(img: Image.Image, filename: str, folder_path: str, file_ext: str = "png", img_type: str = None, quality: float = 100):
    filename = remove_file_ext(filename)

    out_filename = filename
    if img_type != None:
        out_filename = f"{img_type}-{filename}"

    out_folder = folder_path
    if img_type != None:
        # If using img type, save to specific img type folder
        out_folder = f"{folder_path}/{img_type}"

    _save_img_helper(img, out_filename, out_folder, file_ext=file_ext, quality=quality)


def load_imgs(folder_path: str) -> list[tuple[Image.Image]]:
    """ Returns list of PIL images from folder """
    sorted_filenames = sorted(os.listdir(folder_path))

    imgs = []

    for fname in sorted_filenames:
        path = f"{folder_path}/{fname}"

        # Ignore folders
        if not os.path.isfile(path):
            continue

        img = Image.open(path)

        imgs.append((fname, img))

    return imgs


def copy_file(input_path: str, output_path: str) -> None:
    """ Copies file from input path to output path (and creates path if it does not exist) """
    create_path_if_not_exists(output_path)
    shutil.copy(input_path, output_path)


def is_absolute(path: str) -> bool:
    return os.path.isabs(path)


def _cache_path(original_img: Image.Image, salt: str, file_ext: str) -> str:
    byte_input = original_img.tobytes()

    if salt is not None:
        byte_input += salt.encode("utf-8")
    
    hash = sha256(byte_input)

    return f"{cache_folder}{hash.hexdigest()}.{file_ext}"


def is_cached(original_img: Image.Image, salt: str = None, file_ext: str = CACHE_FILE_EXT) -> bool:
    path = _cache_path(original_img, salt, file_ext)
    
    # Important to use isfile for synchronization purposes
    return os.path.isfile(path)


def save_img_cache(original_img: Image.Image, cache_img: Image.Image, salt: str = None, file_ext: str = CACHE_FILE_EXT) -> str:
    """ Saves cache_img to cache, with filename from original_img hash. Returns cached path. """

    # Create cache folder
    create_path_if_not_exists(cache_folder)
    
    # Save to temporary path, and move file to ensure file write is atomic
    tmp_path = f"{cache_folder}/tmp.{file_ext}"
    cache_img.save(tmp_path, format=file_ext)

    cached_path = _cache_path(original_img, salt, file_ext)

    os.replace(tmp_path, cached_path) # Move tmp to correct path

    return cached_path


def load_img_cache(original_img: Image.Image, salt: str = None, file_ext: str = CACHE_FILE_EXT) -> Image.Image:
    """ Returns cached image file, based on original_img hash. Ensure cached image exists before calling """
    cached_path = _cache_path(original_img, salt, file_ext)

    return Image.open(cached_path)