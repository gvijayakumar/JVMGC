# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:27:06 2023

@author: Admin
"""

import pickle

# Data to be saved to the pickle file
data_to_save = {'name': 'John', 'age': 30, 'city': 'New York'}

# Save data to a pickle file
with open('data.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)

# Load data from the pickle file
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)
