

import matplotlib.pyplot as plt

# Labeled data values
labeled_data = [2, 3, 4, 5, 6]

# Model accuracy values
model_accuracy_1 = [0.872, 0.877, 0.873, 0.866, 0.866]
#model_accuracy_2 = [0.912, 0.919, 0.925, 0.929]

xtick_locations = [2, 3, 4, 5, 6]
xtick_labels = ['2', '3', '4', '5','6']
plt.xticks(xtick_locations, xtick_labels)

# Plot the line diagrams
plt.plot(labeled_data, model_accuracy_1, marker='o')
#plt.plot(labeled_data, model_accuracy_2, marker='s', label='Model 2')

# Set the title and labels for the diagram
plt.title('Model Dice Coefficient over Number of weak SVM Annotators')
plt.xlabel('Number of SVM annotators')
plt.ylabel('Dice Coefficient')

# Add a legend to the diagram
plt.legend()

# Show the diagram
plt.show()