from dataset_handler import read_img
import numpy as np
import math

# Extract pixels from inpuyed photo and return them as list
def img_to_vector(path):
    # read image, returns list of lists
    img_matrix = read_img(path)
    # Concatenate all this list into one
    np_vector = np.concatenate(img_matrix)
    img_vector = [i for i in np_vector]
    return img_vector


# Transforms vector into mean deviation form
def to_deviation_form(vector, mean):
    for i in range(len(vector)):
        vector[i] -= mean[i]
    return vector


# Project photo at evc`s space
def project_photo(photo, evcs):
    projected_photo = []
    for i in evcs:
        projected_photo.append(np.inner(i, photo))
    return projected_photo

def distance(v1, v2):
    dist = 0
    for i in range(len(v1)):
        dist += (v1[i] - v2[i]) ** 2
    return math.sqrt(dist)

def nearest_vector(img_vector, photos):
    distances = dict()
    for i in range(len(photos)):
        distances[i] = distance(img_vector, photos[i])
    nearest = min(distances.values())
    for key in distances:
        if distances[key] == nearest:
            return key


if __name__ == "__main__":
    dct = {"1000":10, "200":20, "300":30}
    mini =  min(dct.values())
    for i in dct:
        if dct[i] == mini:
            print(i)

