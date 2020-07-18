# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:23:55 2018

@author: ahmed
"""
import pickle
import numpy as np
#import random
import matplotlib.pyplot as plt
import math
import cv2
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers import (convolution2d, 
    max_pool2d, 
    batch_norm,
    fully_connected,
    xavier_initializer,
    avg_pool2d,
    flatten)
from tqdm import tqdm
from functools import partial
from drawplot import plot_instance_counts
from visual_color import plot_color_statistic
from preprocessData import convert_to_gray 
from preprocessData import normalize_and_center
from preprocessData import create_normalization_function
from sklearn.model_selection import train_test_split

#from showimage import plot_n_images_for_class

training_file = "./traffic-signs-data/train.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']    

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


plot_instance_counts(y_train, 'Training Data')

def indices_for_class(class_label, labels=y_train):
    return np.where(labels == [class_label])[0]

def plot_n_images_for_class(class_label, n=4, dataset_source=training_file, labels=y_train, cmap='jet'):
    with open(dataset_source, mode='rb') as f:
        images = pickle.load(f)['features']
        indices = np.random.choice(indices_for_class(class_label), n)
        figure = plt.figure(figsize = (6,6))

        for i, index in enumerate(indices):
            a = figure.add_subplot(n,n, i+1)
            img = plt.imshow(images[index], interpolation='nearest', cmap=cmap)
            plt.axis('off')
        plt.suptitle('images for class {}'.format(class_label))
        plt.show()
        
for i in np.arange(43):
    plot_n_images_for_class(i)
    
plot_color_statistic(training_file, "Training Data", 1000)
 
# grayscaled data
grayscaled = [convert_to_gray(img) for img in X_train]
# normalizer function based on the data in the training set
normalizer = create_normalization_function(grayscaled)

grayscaled_and_centered = np.asarray([normalizer(img) for img in grayscaled])

grayscaled_training = {
    'features' : grayscaled_and_centered
}
grayscale_file = 'gray_training.p'
pickle.dump(grayscaled_training, open(grayscale_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
del grayscaled_training
del grayscaled_and_centered

plot_color_statistic(grayscale_file, 'Grayscaled and centered data', 10000)

for i in np.random.choice(43, 10):
    plot_n_images_for_class(i, dataset_source=grayscale_file, cmap='gray')
    
# #show image of 10 random data points
#fig, axs = plt.subplots(2,5, figsize=(15, 6))
#fig.subplots_adjust(hspace = .2, wspace=.001)
#axs = axs.ravel()
#for i in range(10):
#    index = random.randint(0, len(X_train))
#    image = X_train[index]
#    axs[i].axis('off')
#    axs[i].imshow(image)
#    axs[i].set_title(y_train[index])
#
## histogram of label frequency
#hist, bins = np.histogram(y_train, bins=n_classes)
#width = 0.7 * (bins[1] - bins[0])
#center = (bins[:-1] + bins[1:]) / 2
#plt.bar(center, hist, align='center', width=width)
#plt.show()

   
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
            
    
#training_file = 'train.p'
#testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_train_final,X_valid_final, y_train_final, y_valid_final = train_test_split(X_train, y_train, stratify=y_train)

print(X_train_final.shape)
print(X_valid_final.shape)
print(y_train_final.shape)
print(y_valid_final.shape)
 
plot_instance_counts(y_train_final, 'Final Training Data')
plot_instance_counts(y_valid_final, 'Final Validation Data')   


final_training_file = 'final_training.p'
final_validation_file = 'final_validation.p'
pickle.dump({
            'features': X_train_final,
            'labels': y_train_final
        }, open(final_training_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump({
            'features': X_valid_final,
            'labels': y_valid_final
        }, open(final_validation_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    
tf.reset_default_graph()
lr = 0.001
sigma = 0.1
hidden = 512
classes= 43
search_epochs = 15
batch_size = 64

def vgg_conv_block(filter_size, inputs):
    k_size = 3
    with (tf.variable_scope("conv_1")):
        conv_1 = convolution2d(
            inputs=inputs,
            num_outputs=filter_size,
            kernel_size=k_size,
            stride=1,
            padding='SAME',
            rate=1,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=sigma),
            biases_initializer=tf.zeros_initializer,
        )
    with (tf.variable_scope("conv_2")):
        conv_2 = layers.convolution2d(
            inputs=conv_1,
            num_outputs=filter_size,
            kernel_size=k_size,
            stride=1,
            padding='SAME',
            rate=1,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=sigma),
            biases_initializer=tf.zeros_initializer,
        )
    return max_pool2d(conv_2, [2,2], [2,2], 'SAME')


x = tf.placeholder(tf.float32, (None, 32, 32))
x_reshaped = tf.reshape(x, (-1, 32, 32, 1))
y = tf.placeholder(tf.float32, (None, classes))

#convolutions
with(tf.variable_scope("vgg_block1")):
    vgg_block_1 = vgg_conv_block(32, x_reshaped)
with(tf.variable_scope("vgg_block2")):
    conv_output = vgg_conv_block(64, vgg_block_1)

#fully connected
fc0 = layers.flatten(conv_output)
fc1 = fully_connected(
    inputs=fc0,
    num_outputs=hidden,
    weights_initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=sigma),
    biases_initializer=tf.zeros_initializer,
    activation_fn=tf.nn.relu
)
keep_prob= tf.placeholder(tf.float32)
fc_dropout = tf.nn.dropout(fc1, keep_prob=keep_prob)
# classifier_head
y_ = fully_connected(
inputs=fc_dropout,
num_outputs=classes,
weights_initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=sigma),
biases_initializer=tf.zeros_initializer,
activation_fn=None
)