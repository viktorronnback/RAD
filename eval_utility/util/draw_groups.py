import os
import cv2

imgs_folder = "../datasets/cityscapes_filtered/images/val"
annotations_folder = "../datasets/cityscapes_filtered/labels/val"

def draw_box(image, box):
    values = box.split(" ")[1:-1]

    x_values = values[::2]
    x_values = [int(float(x) * image.shape[1]) for x in x_values]

    y_values = values[1::2]
    y_values = [int(float(y) * image.shape[0]) for y in y_values]

    cv2.rectangle(image, (min(x_values), min(y_values)), (max(x_values), max(y_values)), color=(0, 0, 255), thickness=2)


if __name__ == "__main__":
    imgs = sorted(os.listdir(imgs_folder))
    labels = sorted(os.listdir(annotations_folder))

    persongroups_found = 0

    for img, label in zip(imgs, labels):
        box = None
        draw = False

        lines = []

        with open(annotations_folder + "/" + label) as f:
            for line in f.readlines():
                if line[0] == "8":
                    persongroups_found += 1
                    draw = True
                    box = line
                    lines.append(line)

        if draw:
            image = cv2.imread(imgs_folder + "/" + img)

            for line in lines:
                draw_box(image, line)

            cv2.imshow("persongroup", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
    print(persongroups_found)

        