# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 22:04:05 2018

@author: ahmed
"""
"""
import cv2
import math
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

class ImageTransformer():
    def rotate_image_random(self, image):
        rotation_range = self.rotation_range
        rotation = np.random.uniform(-rotation_range, rotation_range)
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,1)
        dst = cv2.warpAffine(image,M,(cols,rows))
        return dst

    def translate_image_random(self, image):
        translation_range = self.translation_range
        x_translate = np.random.uniform(0, translation_range)
        y_translate = np.random.uniform(0, translation_range)
        rows, cols = image.shape
        M = np.float32([[1,0,x_translate],[0,1,y_translate]])
        dst = cv2.warpAffine(image,M,(cols,rows))
        return dst

    def shear_image_random(self, image):
        shear_factor_range = self.shear_factor_range
        rows, cols = image.shape
        src = np.float32([[0, 0], [0, cols], [rows, cols], [rows, 0]])
        left_edge_perspective = np.random.uniform((1-shear_factor_range)*cols, (1+shear_factor_range)*cols)
        right_edge_perspective = np.random.uniform((1-shear_factor_range)*cols, (1+shear_factor_range)*cols)
        tranform = np.float32([
                [rows-left_edge_perspective, 0],
                [rows-right_edge_perspective, cols],
                [right_edge_perspective, cols],
                [left_edge_perspective, 0]
            ])
        M = cv2.getPerspectiveTransform(src, tranform)
        dst = cv2.warpPerspective(image,M,(rows, cols))
        return dst


    def zoom_image_random(self, image):
        zoom_range = self.zoom_range
        zoom = np.random.uniform(1-zoom_range, 1+zoom_range)
        rows, cols= image.shape
        new_rows = new_cols = math.floor(zoom * rows)
        if zoom < 1:
            # zoom out
            padding = math.ceil(rows - new_rows)
            dst = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        else:
            # zoom in
            zoomed_image = cv2.resize(image, (new_rows, new_cols))
            crop = math.floor((new_rows - rows) / 2)
            dst = zoomed_image[crop: crop+rows, crop: crop+rows]
        return cv2.resize(dst, (rows, cols)) # ensure size is same as input


    def __init__(self, zoom_range, shear_factor_range, rotation_range, translation_range):
        self.zoom_range = zoom_range
        self.shear_factor_range = shear_factor_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        
    def tranform(self, image):
        zoomed = self.zoom_image_random(image)
        sheared = self.shear_image_random(zoomed)
        rotated = self.rotate_image_random(sheared)
        translated = self.translate_image_random(rotated)
        return translated
    
def batch_generator(arr, batch_size):
    n = len(arr)
    start = range(0, n, batch_size)
    end = range(batch_size, n+batch_size, batch_size)
    for i,j in zip(start, end):
        if j > n:
            yield arr[i:n]
        else:
            yield arr[i:j]
            
    
training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_train_final,X_valid_final, y_train_final, y_valid_final = train_test_split(X_train, y_train, stratify=y_train)

print(X_train_final.shape)
print(X_valid_final.shape)
print(y_train_final.shape)
print(y_valid_final.shape)
"""


            
            

    

