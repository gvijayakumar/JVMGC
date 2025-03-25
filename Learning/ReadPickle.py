# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:18:59 2023

@author: Admin
"""
  
import pickle



def print_model_contents(obj, indent=0):
    if isinstance(obj, (list, tuple, set)):
        for i, item in enumerate(obj):
            print(' ' * indent + f'[{i}]')
            print_model_contents(item, indent + 4)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            print(' ' * indent + f'{key}:')
            print_model_contents(value, indent + 4)
    elif hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            print(' ' * indent + f'{key}:')
            print_model_contents(value, indent + 4)
    else:
        print(' ' * indent + repr(obj))



file_path = r'C:\Users\Admin\Downloads\classifier_mall_cust_1.pkl'

try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print("Data loaded successfully from the pickle file:")
        print(data)
except FileNotFoundError:
    print(f"File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
try:
    with open(file_path, 'rb') as file:
        # Load the content of the pickle file
        data = pickle.load(file)

        # Print all contents of the loaded object
        print_model_contents(data)

except FileNotFoundError:
    print(f"File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")



