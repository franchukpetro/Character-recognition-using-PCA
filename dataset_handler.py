import cv2
import os
import numpy as np


# Reads from directory all images and extracts each pixel from picture.
# Then put all pixels of one picture as a column of matrix.
def dataset_to_matrix(path):
    letters_dict = dict()
    # list of all images in directory
    img_lst = sorted([file for file in os.listdir(path)])

    # create nxm matrix
    n = read_img(path + img_lst[0]).size
    m = len(img_lst)
    img_matrix = np.array([[0 for col in range(m)] for row in range(n)])

    # each image after reading will be represented as array of arrays, so we concatenate them to one
    for img in range(len(img_lst)):

        if img_lst[img][0] not in letters_dict.keys():
            letters_dict[img_lst[img][0]] = [img]
        else:
            letters_dict[img_lst[img][0]].append(img)
        letter = np.concatenate(read_img(path + img_lst[img]))
        for i in range(len(letter)):
            # put pixels of one image into columns of matrix
            img_matrix[i][img] = letter[i]
    return img_matrix, letters_dict


# Extract pixels from image and return array of arrays of pixels
def read_img(path_to_file):
    # read image as grayscale picture
    img = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)

    # convert grayscale to black and white image, to have just one parameter of pixel instead of RGB
    # 0 stands for  black, 255 for white
    (thresh, img_bw) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_bw


def project_dataset(dataset, evcs):
    Y = []
    for photo_idx in range(len(dataset[0])):
        photo_vector = [i[photo_idx] for i in dataset]
        y = []
        for j in evcs:
            y.append(np.inner(j, photo_vector))
        Y.append(y)
    return Y

if __name__ == "__main__":
    path = "./training_dataset/"
    ds =  dataset_to_matrix(path)
    print(ds[1])
