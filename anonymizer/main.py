import threading
import sys

from PIL import Image

from diffusion_pipeline import DiffusionPipeline
from util import cutout, stitch, segment, controlnet_utils, image_utils, file_utils
from util.settings_handler import Settings
from util.safety_checker import SafetyChecker
from util import time_module as tm

def _save_images(img_filename: str, ext: str, original: Image.Image, annot: Image.Image, controlnet: Image.Image, generated: Image.Image, final: Image.Image, settings: Settings) -> None:
    file_utils.save_img(original, img_filename, settings.output, img_type="original", file_ext=ext)
    file_utils.save_img(generated, img_filename, settings.output, img_type="gen", file_ext=ext)
    file_utils.save_img(annot, img_filename, settings.output, img_type="annot", file_ext=ext)

    # Always save controlnet input as PNG (usually smaller sizes for images with few colors)
    if settings.controlnet == "both":
        file_utils.save_img(controlnet[0], img_filename, settings.output, img_type="control-0", file_ext="png")
        file_utils.save_img(controlnet[1], img_filename, settings.output, img_type="control-1", file_ext="png")
        
        collage_images = [original, generated, final, controlnet[0]]
    else:
        file_utils.save_img(controlnet, img_filename, settings.output, img_type="control", file_ext="png")

        collage_images = [original, generated, final, controlnet]
        
    
    file_utils.save_img(final, img_filename, settings.output, img_type="final", file_ext=ext)

    collage = image_utils.create_collage(collage_images)
    file_utils.save_img(collage, img_filename, settings.output, img_type="collage", file_ext=ext)


def _print_info(action: str, fname: str, index: int, num_images: int):
    progress = f"{index} / {num_images}"

    print(f"{action} ({fname}) ({progress})")


def preprocess_images(settings: Settings) -> list[tuple[str, Image.Image]]:
    images = []
    for fname, img in file_utils.load_imgs(settings.input):
        if (img.width < 512 or img.height < 512):
            print(f"WARNING! Stable diffusion handles low resolution ({img.width} x {img.height} < 512 x 512) images poorly, consider using a super resolution tool such as Real-ESRGAN")
        
        img = image_utils.preprocess_image(img, settings.max_output_width, settings.max_output_height)
        
        images.append((fname, img))
    
    return images


def _output_file_ext(settings: Settings, index: int, fname: str) -> tuple[str, str]:
    """ Returns tuple of output name and ext """
    
    if settings.save_format == "default":
            ext = fname.split(".")[-1]
    else:
        ext = settings.save_format
    
    output_fname = f"{index}.{ext}"
    
    return (output_fname, ext)


def anonymize(settings: Settings, safety_checker: SafetyChecker, images: list[Image.Image], diff_pipeline: DiffusionPipeline, init_time: float) -> None:
    num_images = len(images)

    for index, (fname, original_image) in enumerate(images):
        gen_start_time = tm.start_time()
        index = index + 1

        # File extension / output filename
        output_fname, ext = _output_file_ext(settings, index, fname)

        # TODO Overwrite functionality is not robust, better would be to cache hashed original images connected to a specific output folder
        action = "Generating"
        if file_utils.path_exists(f"{settings.output}/final/final-{output_fname}"):
            if not settings.overwrite:
                _print_info("Image already generated, skipping", fname, index, num_images)
                continue
            else:
                action = "Image already generated, overwriting"

        _print_info(action, fname, index, num_images)
                        
        if not settings.seg_parallel:
            t0 = tm.start_time()
            seg_msg = segment.generate_annotation(original_image, fname, settings.max_output_width, settings.max_output_height, settings)
            tm.print_elapsed_seconds(settings, t0, "Detection + Segmentation")
            print(seg_msg)

        annot_image = segment.wait_for_annotated(img=original_image, fname=fname)

        # TODO Maybe save empty image to cache, so YOLO does not have to be re-run for same image 
        if annot_image == None:
            _print_info("Image does not contain any people, saving copy of original", fname, index, num_images)
            file_utils.save_img(original_image, output_fname, settings.output, file_ext=ext, img_type="final")
            continue

        t1 = tm.start_time()
        # Cut out image
        cutout_image = cutout.gen_cutout(original=original_image, mask=annot_image)
        tm.print_elapsed_seconds(settings, t1, "Cut out")


        # Canny / OpenPose
        canny_input = annot_image if settings.canny_silhouette else cutout_image

        if settings.controlnet == "canny":
            input_controlnet = controlnet_utils.gen_canny(canny_input, settings.canny_min_threshold, settings.canny_max_threshold) # Canny
        elif settings.controlnet == "pose":
            input_controlnet = controlnet_utils.gen_pose(cutout_image) # OpenPose
        elif settings.controlnet == "both":
            t2 = tm.start_time()
            input_controlnet = [controlnet_utils.gen_canny(canny_input, settings.canny_min_threshold, settings.canny_max_threshold)] # Canny
            tm.print_elapsed_seconds(settings, t2, "Canny extraction")

            t3 = tm.start_time()
            input_controlnet.append(controlnet_utils.gen_pose(cutout_image)) # OpenPose
            tm.print_elapsed_seconds(settings, t3, "Pose extraction")

        # Generate
        t4 = tm.start_time()
        generated = diff_pipeline.generate(input_image=original_image, input_controlnet=input_controlnet)
        tm.print_elapsed_seconds(settings, t4, "Diffusion")

        t5 = tm.start_time()
        if not safety_checker.is_safe(generated):
            print("Generated image potentially contains NSFW content, not saving. Consider changing model/prompt.")
            continue
        
        tm.print_elapsed_seconds(settings, t5, "Safety checker")


        # Save cutout of generated
        generated_cutout = cutout.gen_cutout(original=generated, mask=annot_image)
        file_utils.save_img(generated_cutout, fname, settings.output, file_ext="jpg", img_type="cutout")

        t6 = tm.start_time()
        # Stitch
        final_image = stitch.stitch_images(foreground_image=generated, background_image=original_image, mask_image=annot_image)
        tm.print_elapsed_seconds(settings, t6, "Stitch")
        
        # Save images
        _save_images(output_fname, ext, original_image, annot_image, input_controlnet, generated, final_image, settings)
        print(f"Generated image in {tm.elapsed_minutes(gen_start_time)} minutes")

        tm.print_average_measurements(settings)

        print(f"Total elapsed time: {tm.elapsed_minutes(init_time)} minutes")

        print("\n")


def main(config_fname: str):
    init_time = tm.start_time()
    
    settings = Settings(config_fname)

    if settings.overwrite and file_utils.path_exists(settings.output):  
        print(f"WARNING! Overwriting output folder {settings.output}")

    # TODO If overwrite is false and folder already exists: check that config.yaml files match, otherwise throw error
    settings.save_copy_to_output()
    
    diff_pipeline = DiffusionPipeline(settings)

    safety_checker = SafetyChecker(settings)

    images = preprocess_images(settings)

    if settings.seg_parallel:
        # Start generating segmentations async
        # TODO separate class for thread, see: https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread
        gen_annot_thread = threading.Thread(target=segment.generate_all_annotations, args=(images, settings.max_output_width, settings.max_output_height, settings))
        gen_annot_thread.start()

    tm.print_elapsed_seconds(settings, init_time, "Initialization")

    anonymize(settings, safety_checker, images, diff_pipeline, init_time)

    if settings.seg_parallel:
        gen_annot_thread.join()

    tm.print_average_measurements(settings)

    tm.print_average_measurements_skip_first(settings)

    print(f"\n\nFinished in {tm.elapsed_minutes(init_time)} minutes")


if __name__ == "__main__":
    # First argument is config filename (defaults to config.yaml)
    config_fname = "config.yaml" if len(sys.argv) < 2 else sys.argv[1]

    main(config_fname)
