# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:27:42 2023

@author: ypatia
"""

import matplotlib.pyplot as plt

# Labeled data values
labeled_data = [1000, 2000,3000, 4000,5000, 10000]

# Model accuracy values
model_accuracy = [0.82,0.87,0.90,0.90,0.91,0.91]

xtick_locations = [1000, 2000,3000, 4000,5000,6000,7000,8000,9000, 10000]
xtick_labels = ['1000', '2000','3000','4000','5000','6000','7000','8000','9000','10000']
plt.xticks(xtick_locations, xtick_labels)
# Plot the line diagram
plt.plot(labeled_data, model_accuracy, marker='o')

# Set the title and labels for the diagram
plt.title('Model Accuracy over Labeled Data')
plt.xlabel('Labeled Data')
plt.ylabel('Model Accuracy')

# Show the diagram
plt.show()