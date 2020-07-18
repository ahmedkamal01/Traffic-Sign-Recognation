
import cv2
from functools import partial
import numpy as np
import pickle
# convert the image from RGB space to gray scale
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# function that subtracts mean and divides by standard deviation of a dataset for each image
def normalize_and_center(image, std, mean):
    return (image - mean) / std

# function which takes a dataset to create the respective normalization function
def create_normalization_function(image_dataset):
    dataset_mean = np.mean(image_dataset)
    dataset_std = np.std(image_dataset)
    print("dataset mean: {}, std: {}".format(dataset_mean, dataset_std))
    stats_file = 'dataset_stats.p'
    pickle.dump({
            'mean': dataset_mean,
            'std': dataset_std
        }, open(stats_file, 'wb'))
    return partial(normalize_and_center, std=dataset_std, mean=dataset_mean)