from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import verification

def random_color():
    return list(np.random.random(size=3) * 256)

def draw_box(image, area, color):
    thickness = 2

    s = (area["x"], area["y"])
    e = (area["x"] + area["w"], area["y"] + area["h"])

    cv2.rectangle(image, s, e, color, thickness) 

def save_facepair(save_path, im1, im2, a1, a2):
    roi1 = im1[a1["y"]:a1["y"]+a1["h"], a1["x"]:a1["x"]+a1["w"]]
    roi2 = im2[a2["y"]:a2["y"]+a2["h"], a2["x"]:a2["x"]+a2["w"]]

    h = min(roi1.shape[0], roi2.shape[0])
    roi1_resized = cv2.resize(roi1, (int(roi1.shape[1] * h / roi1.shape[0]), h))
    roi2_resized = cv2.resize(roi2, (int(roi2.shape[1] * h / roi2.shape[0]), h))

    facepair = np.concatenate((roi1_resized, roi2_resized), axis=1)

    cv2.imwrite(save_path, facepair)
    print("Saved recognized face pair to", save_path)


def verify_images(img1_path: str, img2_path: str, save_eval: bool=False):
    """ Verifies the distance between all pairs of faces in two images. 

    Arguments:
        img1_path, img2_path (str, str): paths to the images. 
        save_eval (bool): whether to save evaluation images, with green/red boxes around recognitions. 
    Returns:
        'results' (dict): A dictionary with the number of "detected_faces", a distance list for faces and recognition metrics.  
    """
    result = verification.verify(img1_path, img2_path, detector_backend="retinaface", distance_metric="cosine", model_name="Facenet512", enforce_detection=False)

    FP = 0 # False positives: number of faulty recognitions.
    TP = 0 # True positives: number of accurate recognitions -> times anonymization failed. 
    FN = 0 # False negatives: number of missed recognitions -> times anonymization was sucessful. 
    TN = 0 # True negatives: number of correct non-recognitions.  

    distances = []

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    eval_path = img1_path.replace("original", "eval")
    recognition_path = img1_path.replace("original/", "eval/recognitions/")
    recognition_path = recognition_path.replace("original-", "tp-")

    for idx, distance in enumerate(result["all_distances"]):
        img1_region, img2_region = result["all_regions"][idx]

        if len(result["all_distances"]) == 1:
            pass # Only 1 face in the image, it has to be the same person.
        elif abs(img2_region["x"] - img1_region["x"]) > 30 or abs(img2_region["y"] - img1_region["y"]) > 30: # Consider bounding boxes in roughly the same pos as same person. 
            if distance < result["threshold"]:
                FP += 1 # Different people recognized as same. 
            else:
                TN += 1 # Different people recognized as different. 
            continue 

        # Save distances for all anonymizations. 
        distances.append(distance)

        # Bounding box color.
        color = (0, 0, 255)       
        
        if distance < result["threshold"]:
            TP += 1 # Original and generated person recognized as same. 
            if save_eval:
                save_facepair(recognition_path.replace("tp-", f"tp-{idx}-"), img1, img2, img1_region, img2_region)
            draw_box(img1, img1_region, color)
            draw_box(img2, img2_region, color)

        else:
            FN += 1 # Original and generated person recognized as different. 
            color = (0, 255, 0)
            draw_box(img1, img1_region, color)
            draw_box(img2, img2_region, color)
            
        text_loc1 = (img1_region["x"], max(img1_region["y"] - 10, 30))
        text_loc2 = (img2_region["x"], max(img2_region["y"] - 10, 30))

        cv2.putText(img1, f"id: {idx}", text_loc1, 2, 1, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img2, f"id: {idx}, d: {np.round(distance, 4)}", text_loc2, 2, 1, color, thickness=2, lineType=cv2.LINE_AA)

    result = {"detected_faces": result["detected_faces"],
              "distances": distances, 
              "FP": FP,
              "TP": TP,
              "FN": FN,
              "TN": TN}
    
    if save_eval:
        try:
            vertical_concat = np.concatenate((img1, img2), axis=0) 

            cv2.imwrite(eval_path, vertical_concat)
            print("Saved eval image to:", eval_path)
        except:
            raise RuntimeError("Eval could not be saved, likely due to image size:", img1.shape)

    return result

def verify_folder(folder: str, save_eval: bool=False):
    """ Verifies faces for every pair of images in subfolders /original and /final. 

    Arguments:
        folder (str): Name of folder with assumed subfolders /original and /final.
    Returns:
        'total' (dict): Dictionary of mismatches, missed faces, FP, TP, FN and TN.  
    """
    original_folder = folder + "/original/"
    final_folder = folder + "/final/"
    eval_folder = folder + "/eval/"
    recognitions_folder = eval_folder + "recognitions/"

    if save_eval and not os.path.exists(recognitions_folder):
        os.makedirs(recognitions_folder)

    # Sort folders.
    original_sorted = sorted((f for f in os.listdir(original_folder) if not f.startswith(".")), key=str.lower)
    final_sorted = sorted((f for f in os.listdir(final_folder) if not f.startswith(".")), key=str.lower)

    distances = np.array([])

    total = {"mismatches": 0,
             "missed_faces": 0,
              "FP": 0,
              "TP": 0,
              "FN": 0,
              "TN": 0
             }
    
    for idx, (original_filename, final_filename) in enumerate(zip(original_sorted, final_sorted)):
        print("\n", idx, "/", len(original_sorted))
        img1_path = original_folder + original_filename
        img2_path = final_folder + final_filename

        result = verify_images(img1_path, img2_path, save_eval)

        original_detections, generated_detections = result["detected_faces"]

        if original_detections != generated_detections:
            total["mismatches"] += 1
            print("Mismatch detected, faces in original/generated image:", result["detected_faces"])

        for k, v in result.items():
            if k in total:
                total[k] += v

        for d in result["distances"]:
            distances = np.append(distances, d)

        print(original_filename, final_filename)
        print(result)
        print("total:", total)

    if save_eval:
        # Plotting histogram over distances. 
        plt.hist(distances[distances != 0], bins=50, color='skyblue', edgecolor='black')
        plt.title('Histogram of Distances for Recognition Model.')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(eval_folder + "/distance_histogram.png")
        plt.show()
    
    total["missed_faces"] = (distances <= 0.1).sum()

    return total


if __name__ == "__main__":
    res = verify_folder("../anonymizer/output/privacy-eval-full", save_eval=True)
    print(res)