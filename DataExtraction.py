# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:53:01 2023

@author: Admin
"""

import re
import csv

# Sample data
data = """
[0.092s][info][gc,heap     ] GC(0) DefNew: 512K(576K)->64K(576K) Eden: 512K(512K)->0K(512K) From: 0K(64K)->64K(64K)
[0.092s][info][gc,heap     ] GC(0) Tenured: 0K(1408K)->367K(1408K)
[0.092s][info][gc,metaspace] GC(0) Metaspace: 4229K(4352K)->4229K(4352K) NonClass: 3908K(3968K)->3908K(3968K) Class: 321K(384K)->321K(384K)
[0.166s][info][gc,heap     ] GC(1) DefNew: 576K(576K)->64K(576K) Eden: 512K(512K)->0K(512K) From: 64K(64K)->64K(64K)
[0.166s][info][gc,heap     ] GC(1) Tenured: 367K(1408K)->515K(1408K)
[0.166s][info][gc,metaspace] GC(1) Metaspace: 5877K(6016K)->5877K(6016K) NonClass: 5380K(5504K)->5380K(5504K) Class: 496K(512K)->496K(512K)
"""

# Regular expression pattern for extracting relevant information
pattern = r'\[([\d.]+)s\]\[info\]\[gc(?:,[^\]]+)?\] ([^\]]+) ([^\s]+)\(([^\)]+)\)->([^\s]+)\(([^\)]+)\)'

# Extracting features using regular expression
matches = re.findall(pattern, data)

# Displaying extracted features
for match in matches:
    timestamp, gc_type, memory_type, before, after, capacity = match
    print(f"Timestamp: {timestamp}, GC Type: {gc_type}, Memory Type: {memory_type}, Before: {before}, After: {after}, Capacity: {capacity}")

# Writing extracted features to CSV
csv_file_path = 'gc_data.csv'
header = ['Timestamp', 'GC Type', 'Memory Type', 'Before', 'After', 'Capacity']

with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    csv_writer.writerows(matches)

print(f'Data has been written to {csv_file_path}')