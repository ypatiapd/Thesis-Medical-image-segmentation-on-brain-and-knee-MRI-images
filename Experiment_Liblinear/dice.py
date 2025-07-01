# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:30:15 2023

@author: ypatia
"""

import numpy as np

# Load the true and predicted labels as 1D lists
true_labels = np.load('true_labels.npy')
predicted_labels = np.load('predicted_labels.npy')

# Calculate the confusion matrix for each class
num_classes = 3
cm = np.zeros((num_classes, num_classes))
for i in range(num_classes):
    for j in range(num_classes):
        cm[i,j] = np.sum((true_labels == i+1) & (predicted_labels == j+1))

# Calculate the Dice coefficient for each class
dice_coeffs = np.zeros((num_classes,))
for i in range(num_classes):
    tp = cm[i,i]
    fp = np.sum(cm[:,i]) - tp
    fn = np.sum(cm[i,:]) - tp
    dice_coeffs[i] = 2*tp / (2*tp + fp + fn)

# Calculate the mean Dice coefficient across all classes
mean_dice_coeff = np.mean(dice_coeffs)