# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:01:32 2023

@author: ypatia
"""

import matplotlib.pyplot as plt

# List of dictionaries representing feature vectors
feature_vectors = [{1: 0.1, 2: 0.5, 3: 0.3, 7: 10, 8: 20, 9: 100},
                   {1: 0.2, 2: 0.4, 3: 0.6, 7: 30, 8: 40, 9: 100},
                   {1: 0.4, 2: 0.8, 3: 0.9, 7: 50, 8: 60, 9: 200},
                   {1: 0.3, 2: 0.7, 3: 0.1, 7: 70, 8: 80, 9: 100},
                   {1: 0.5, 2: 0.2, 3: 0.4, 7: 90, 8: 100, 9: 200}]

# List of labels
labels = [1, 0, 2, 1, 2]

# Select only instances where z=100
selected_feature_vectors = [fv for fv in feature_vectors if fv.get(9) == 100]

# Get x and y coordinates from feature vectors
x_coords = [fv.get(7) for fv in selected_feature_vectors]
y_coords = [fv.get(8) for fv in selected_feature_vectors]

# Create scatter plot
plt.scatter(x_coords, y_coords, c=labels)
plt.show()
This code will only select the instances where z

network error



