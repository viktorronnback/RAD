# Adapted from https://github.com/serengil/deepface/tree/master

# built-in dependencies
import time
from typing import Any, Dict, Union

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.modules import representation, detection, modeling, verification
from deepface.models.FacialRecognition import FacialRecognition

def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
) -> Dict[str, Any]:
    """
    Verify if an image pair represents the same person or different persons.

    The verification function converts facial images to vectors and calculates the similarity
    between those vectors. Vectors of images of the same person should exhibit higher similarity
    (or lower distance) than vectors of images of different persons.

    Args:
        img1_path (str or np.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        img2_path (str or np.ndarray): Path to the second image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv)

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)

    Returns:
        result (dict): A dictionary containing verification results.

        - 'verified' (bool): Indicates whether the images represent the same person (True)
            or different persons (False).

        - 'distance' (float): The distance measure between the face vectors.
            A lower distance indicates higher similarity.

        - 'max_threshold_to_verify' (float): The maximum threshold used for verification.
            If the distance is below this threshold, the images are considered a match.

        - 'model' (str): The chosen face recognition model.

        - 'similarity_metric' (str): The chosen similarity metric for measuring distances.

        - 'facial_areas' (dict): Rectangular regions of interest for faces in both images.
            - 'img1': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the first image.
            - 'img2': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the second image.

        - 'time' (float): Time taken for the verification process in seconds.
    """

    tic = time.time()

    # --------------------------------
    model: FacialRecognition = modeling.build_model(model_name)
    target_size = model.input_shape

    # img pairs might have many faces
    img1_objs = detection.extract_faces(
        img_path=img1_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )

    img2_objs = detection.extract_faces(
        img_path=img2_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )
    # --------------------------------
    distances = []
    regions = []
    # now we will find the face pair with minimum distance
    for img1_obj in img1_objs:
        img1_content = img1_obj["face"]
        img1_region = img1_obj["facial_area"]
        for img2_obj in img2_objs:
            img2_content = img2_obj["face"]
            img2_region = img2_obj["facial_area"]
            img1_embedding_obj = representation.represent(
                img_path=img1_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img2_embedding_obj = representation.represent(
                img_path=img2_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img1_representation = img1_embedding_obj[0]["embedding"]
            img2_representation = img2_embedding_obj[0]["embedding"]

            if distance_metric == "cosine":
                distance = verification.find_cosine_distance(img1_representation, img2_representation)
            elif distance_metric == "euclidean":
                distance = verification.find_euclidean_distance(img1_representation, img2_representation)
            elif distance_metric == "euclidean_l2":
                distance = verification.find_euclidean_distance(
                    verification.l2_normalize(img1_representation), verification.l2_normalize(img2_representation)
                )
            else:
                raise ValueError("Invalid distance_metric passed - ", distance_metric)

            distances.append(distance)
            regions.append((img1_region, img2_region))

    # -------------------------------
    threshold = verification.find_threshold(model_name, distance_metric)
    distance = min(distances)  # best distance
    facial_areas = regions[np.argmin(distances)]

    toc = time.time()

    # pylint: disable=simplifiable-if-expression
    resp_obj = {
        "verified": True if distance <= threshold else False,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "detector_backend": detector_backend,
        "similarity_metric": distance_metric,
        "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]},
        "time": round(toc - tic, 2),
        "all_distances": distances,
        "all_regions": regions,
        "detected_faces": (len(img1_objs), len(img2_objs))
    }

    return resp_obj