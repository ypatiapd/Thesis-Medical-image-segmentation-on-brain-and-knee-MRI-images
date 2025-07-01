# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:34:33 2023

@author: ypatia
"""

for j in range(0,6):
    for i, instance in enumerate(lab_x_all[j]):
            for feature, value in instance.items():
                if np.isnan(value):
                    print(f"NaN value found in instance {i}, feature {feature}")