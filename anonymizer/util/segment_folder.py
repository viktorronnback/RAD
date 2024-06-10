import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import os.path

import segment

# Function from official demo code in Segment anything. https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def _show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def _generate_annotations(image_folder: str, individual_masks: bool=False, save=True, show=False):
    """ Generates an annotations folder next to the imgs folder containing segmentations 
    of all people in the image.  

    image_folder - assumes the follwing format: "<some-path-to-dir>/<image_category>/imgs"
    """
    destination_folder = image_folder.replace("imgs", "annotations")

    if not os.path.exists(destination_folder) and save:
        os.makedirs(destination_folder)

    # TODO Use generate_annotation() from segment.py

    # Load detection model  
    detection_model = segment.load_YOLO()

    # Load segmentation model.
    sam, predictor = segment.load_SAM()

    for image_filename in os.listdir(image_folder):
        destination_path = destination_folder + "/" + image_filename
        image_path = image_folder + "/" + image_filename
        
        # Run person detection (class=0) with YOLO model. 
        results = detection_model.predict(image_path, save=False, classes=0)
        boxes = None

        for r in results:
            boxes = r.boxes.xyxy

        # Load image.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        # Transform boxes into SAM's expected input.
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes, image.shape[:2])

        # Run SAM model on all the boxes.
        predictor.set_image(image)
        masks, scores, logits = predictor.predict_torch(
            boxes=transformed_boxes,
            multimask_output=False,
            point_coords=None,
            point_labels=None
        )

        # Save each individual mask to file. 
        if individual_masks:
            masks_folder = destination_folder + "/" + image_filename.rstrip(".jpg")
            if not os.path.exists(masks_folder):
                os.makedirs(masks_folder)
            for i in range(len(masks) - 1):
                binary_mask = masks[i][0].numpy().astype(int)
                mask_image = (binary_mask * 255).astype(np.uint8)
                mask_name = image_filename.replace(".jpg", f"{i}.jpg")  
                mask_path = destination_path.replace(".jpg", f"/{mask_name}") 
                cv2.imwrite(mask_path, mask_image)

        # Combine all masks into one. 
        final_mask = None
        for i in range(len(masks) - 1):
            if final_mask is None:
                final_mask = np.bitwise_or(masks[i][0], masks[i+1][0])
            else:
                final_mask = np.bitwise_or(final_mask, masks[i+1][0])

        # Save binary mask to file.
        if save:
            binary_mask = np.where(final_mask > 0.5, 1, 0)
            mask_image = (binary_mask * 255).astype(np.uint8)  
            cv2.imwrite(destination_path, mask_image)

        # Alternatively, show mask on image
        if show:
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            _show_mask(final_mask, plt.gca())
            plt.axis('on')
            plt.show()  

if __name__ == "__main__":
    # TODO argparse input folder
    # input_folder = "../datasets/MOT17/train/MOT17-11-FRCNN/img1/"
    input_folder = "input/MOT17-11-FRCNN/imgs"

    _generate_annotations(input_folder, individual_masks=False, save=True, show=False)