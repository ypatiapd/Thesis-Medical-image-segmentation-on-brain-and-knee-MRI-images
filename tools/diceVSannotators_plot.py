# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:35:46 2023

@author: ypatia
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:27:42 2023

@author: ypatia
"""

import matplotlib.pyplot as plt

# Labeled data values
annotators = [2,3,4,5,6]

# Model accuracy values
Dice_coef = [0.8596,0.8657,0.8620,0.8488,0.8460]

xtick_locations = [2, 3,4, 5,6]
xtick_labels = ['2','3','4','5','6']
plt.xticks(xtick_locations, xtick_labels)
# Plot the line diagram
plt.plot(annotators, Dice_coef, marker='o')

# Set the title and labels for the diagram
plt.title('Model Dice Coefficient over Number of SVM weak Annotators')
plt.xlabel('Number of SVM Annotators')
plt.ylabel('Dice Coefficient')

# Show the diagram
plt.show()