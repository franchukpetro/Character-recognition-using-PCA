import numpy as np
import os
from dataset_handler import dataset_to_matrix, project_dataset
from input_photo import img_to_vector, to_deviation_form, project_photo, nearest_vector
from covariance_matrix import to_mean_deviation_form, cov_matrix
from eigenvectors import eiginevectors, map_back


def main():
    # Get data ready
    ds = dataset_to_matrix("./training_dataset/")
    A = ds[0]
    dictionary = ds[1]

    # Find covariance matrix
    mean = np.mean(A, axis=1)
    B = to_mean_deviation_form(A, mean)
    cov_mat = cov_matrix(B)

    # Find mapped evc
    evcs = eiginevectors(cov_mat)
    mapped_evcs = map_back(A, evcs)

    # Project dataset
    projected_dataset = project_dataset(A, mapped_evcs)

    for letter in  sorted([letters for letters in os.listdir("./input_img/")]):
        imgs = sorted([file1 for file1 in os.listdir("./input_img/" + letter)])

        for img in imgs:
            total_recognitions = len(imgs)
            correct_recognitions = 0

            # Get photo ready
            input_img = img_to_vector("./input_img/" + letter + "/" + img)
            normalized_img = to_deviation_form(input_img, mean)

            # Project photo
            projected_img = project_photo(normalized_img, mapped_evcs)

            # Find nearest photo
            result = nearest_vector(projected_img, projected_dataset)

            for key, values in dictionary.items():
                if result in values:
                    if key == letter:
                        correct_recognitions += 1
        print("Testting of letter" + letter + "finished")
        print("Total: " + str(total_recognitions))
        print("Correct: " + str(correct_recognitions))


if __name__ == "__main__":
    main()