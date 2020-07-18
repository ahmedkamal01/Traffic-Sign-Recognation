# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:17:40 2018

@author: ahmed
"""
#import matplotlib.pyplot as plt
#import numpy as np
#def indices_for_class(class_label, labels=y_train):
#    return np.where(labels == [class_label])[0]
#
#def plot_n_images_for_class(class_label, n=4, dataset_source=training_file, labels=y_train, cmap='jet'):
#    with open(dataset_source, mode='rb') as f:
#        images = pickle.load(f)['features']
#        indices = np.random.choice(indices_for_class(class_label), n)
#        figure = plt.figure(figsize = (6,6))
#
#        for i, index in enumerate(indices):
#            a = figure.add_subplot(n,n, i+1)
#            img = plt.imshow(images[index], interpolation='nearest', cmap=cmap)
#            plt.axis('off')
#        plt.suptitle('images for class {}'.format(class_label))
#        plt.show()
