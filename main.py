import numpy as np
from dataset_handler import dataset_to_matrix, project_dataset
from input_photo import img_to_vector, to_deviation_form, project_photo, nearest_vector
from covariance_matrix import to_mean_deviation_form, cov_matrix
from eigenvectors import eiginevectors, map_back


def main(path_to_photo):
    # Get data ready
    ds = dataset_to_matrix("./small_dataset/")
    A = ds[0]
    dictionary = ds[1]

    # Find covariance matrix
    mean = np.mean(A, axis=1)
    B = to_mean_deviation_form(A, mean)
    cov_mat = cov_matrix(B)

    # Find mapped evc
    evcs = eiginevectors(cov_mat)
    mapped_evcs = map_back(A, evcs)

    # Get photo ready
    input_img = img_to_vector(path_to_photo)
    normalized_img = to_deviation_form(input_img, mean)

    # Project dataset
    projected_dataset = project_dataset(A, mapped_evcs)

    # Project photo
    projected_img = project_photo(normalized_img, mapped_evcs)

    # Find nearest photo
    result = nearest_vector(projected_img, projected_dataset)

    for key, values in dictionary.items():
        if result in values:
            return key

