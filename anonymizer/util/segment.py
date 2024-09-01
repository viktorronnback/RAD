from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
from PIL import Image
import torch

import time

from util import image_utils
from util import file_utils
from util.settings_handler import Settings
from util.time_module import start_time, print_elapsed_seconds

MAX_WAIT_FOR_ANNOTATION_SEC = 120

skipped_img_fnames = [] # Shared between main thread and segmentation

def load_YOLO(device: str, model: str):
    # There are many different sizes for the pretrained YOLOv8-model. Yolov8x is the largest one. 
    yolo = YOLO(model)

    yolo.to(device)

    return yolo


def load_SAM(device: str, model: str):
    # There are three different types of the SAM-model, currently using default.
    sam = sam_model_registry["default"](checkpoint=model)
    sam.to(device)
    predictor = SamPredictor(sam)
    return sam, predictor


def _gen_det_boxes(detection_model: YOLO, img: Image.Image) -> torch.Tensor:
    """ Generates object detection boxes in SAM format """
    # TODO Add more classes like backpacks
    # 0 corresponds to person class
    detections = detection_model.predict(img, save=False, classes=0)
    boxes = None

    for r in detections:
        boxes = r.boxes.xyxy
    
    return boxes


def _gen_seg_masks(predictor: SamPredictor, boxes: torch.Tensor, img: Image.Image) -> torch.Tensor:
    # Transform boxes into SAM's expected input.
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes, img.shape[:2])

    # Run SAM model on all the boxes.
    predictor.set_image(img)
    masks, scores, logits = predictor.predict_torch(
        boxes=transformed_boxes,
        multimask_output=False,
        point_coords=None,
        point_labels=None
    )

    masks = masks.cpu() # Copy tensor from GPU to CPU

    return masks


def _annotate(img: Image.Image, settings: Settings) -> Image.Image | None:
    """ Generates annotation image (segmentation) for the given image. 
    
    Arguments:
        image: image to segment. 
    Returns:
        mask_image: annotated segment(s) as a binary image or None if no people in img.
    """
    # Convert image to RGB
    img = img.convert("RGB")

    detection_model = load_YOLO(settings.seg_device, settings.yolo_obj_det_model)
    sam, predictor = load_SAM(settings.seg_device, settings.sam_seg_model)
    
    t0 = start_time()
    boxes = _gen_det_boxes(detection_model=detection_model, img=img)
    print_elapsed_seconds(settings, t0, "Detection")

    if boxes.numel() == 0:
        # No detections
        return None
    
    t1 = start_time()
    masks = _gen_seg_masks(predictor=predictor, boxes=boxes, img=img)

    # Combine all masks into one. 
    final_mask = masks[0][0]
    for i in range(len(masks) - 1):
        final_mask = np.bitwise_or(final_mask, masks[i+1][0])

    binary_mask = np.where(final_mask > 0.5, 1, 0)
    mask_image = (binary_mask * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_image)

    print_elapsed_seconds(settings, t1, "Segmentation")
    
    return mask_image


def wait_for_annotated(img: Image.Image, fname: str) -> Image.Image | None:
    start_time = time.time()
    
    while not file_utils.is_cached(img):
        # No need for sync here since skipped is checked until it times out
        if fname in skipped_img_fnames:
            return None

        time.sleep(0.5) # Sleep for some time to not spam check

        if time.time() - start_time > MAX_WAIT_FOR_ANNOTATION_SEC:
            raise TimeoutError(f"No annotation in > {MAX_WAIT_FOR_ANNOTATION_SEC}s")
        pass

    return file_utils.load_img_cache(img)


def generate_annotation(img: Image.Image, fname: str, max_width: int, max_height: int, settings: Settings) -> str | bool:
    img = image_utils.preprocess_image(img, max_width, max_height)
    
    if not file_utils.is_cached(img):
        # Generate new annotation if not previously created
        annot = _annotate(img, settings)

        if annot == None:
            skipped_img_fnames.append(fname)
            return f"\nSkipped segmentation {fname}, no people detected"

        annot_path = file_utils.save_img_cache(original_img=img, cache_img=annot)
        return f"\nGenerated segmentation for {fname} as {annot_path}"
    else:
        return f"Segmentation already generated ({fname})"


def generate_all_annotations(images: list[tuple[str, Image.Image]], max_width: int, max_height: int, settings: Settings):
    """ Generates all annotations for given image paths """

    num_images = len(images)

    for index, (fname, img) in enumerate(images):
        # Resize image
        progress = f"({index+1} / {num_images})"

        annot_msg = generate_annotation(img, fname, max_width, max_height, settings)

        print(f"{annot_msg} {progress}")
    
    print(f"All segmentations generated / found ({num_images} / {num_images})")
