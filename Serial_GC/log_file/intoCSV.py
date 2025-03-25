import os
import re
import csv

def extract_features(log_file):
    features = []
    with open(log_file, 'r') as file:
        for line in file:
            # Regular expression patterns to match different types of GC events and relevant information
            gc_pattern = re.compile(r'GC\((\d+)\) (\w+) \(([^)]+)\)')
            heap_pattern = re.compile(r'Heap .*: (\d+)K\((\d+)K\)->(\d+)K\((\d+)K)')
            metaspace_pattern = re.compile(r'Metaspace: (\d+)K\((\d+)K)->(\d+)K\((\d+)K)')
            
            # Match patterns in the log line
            gc_match = gc_pattern.search(line)
            heap_match = heap_pattern.search(line)
            metaspace_match = metaspace_pattern.search(line)
            
            # Extract features from the matched patterns
            if gc_match:
                gc_number = gc_match.group(1)
                gc_type = gc_match.group(2)
                pause_duration = gc_match.group(3)
                features.append({'GC_Number': gc_number, 'GC_Type': gc_type, 'Pause_Duration': pause_duration})
            elif heap_match:
                initial_heap, max_heap, current_heap, max_heap_after_gc = heap_match.groups()
                features[-1]['Initial_Heap'] = initial_heap
                features[-1]['Max_Heap'] = max_heap
                features[-1]['Current_Heap'] = current_heap
                features[-1]['Max_Heap_After_GC'] = max_heap_after_gc
            elif metaspace_match:
                initial_metaspace, max_metaspace, current_metaspace, max_metaspace_after_gc = metaspace_match.groups()
                features[-1]['Initial_Metaspace'] = initial_metaspace
                features[-1]['Max_Metaspace'] = max_metaspace
                features[-1]['Current_Metaspace'] = current_metaspace
                features[-1]['Max_Metaspace_After_GC'] = max_metaspace_after_gc

    return features

def write_to_csv(features, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['GC_Number', 'GC_Type', 'Pause_Duration', 'Initial_Heap', 'Max_Heap', 'Current_Heap', 'Max_Heap_After_GC', 'Initial_Metaspace', 'Max_Metaspace', 'Current_Metaspace', 'Max_Metaspace_After_GC']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for feature in features:
            writer.writerow(feature)

def process_all_logs_in_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    print("Found files:", files)  # Print list of files found in the folder
    # Filter out files with any extension
    log_files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]
    print("Log files to process:", log_files)  # Print list of log files to be processed

    # Process each log file
    for log_file in log_files:
        log_file_path = os.path.join(folder_path, log_file)
        print("Processing:", log_file_path)  # Print path of the log file being processed
        # Generate output CSV file name based on log file name
        output_csv = os.path.splitext(log_file)[0] + '_features.csv'
        # Extract features from the log file
        features = extract_features(log_file_path)
        # Write features to a CSV file
        write_to_csv(features, os.path.join(folder_path, output_csv))
        print("CSV file written:", output_csv)  # Print path of the CSV file written

# Specify the folder path containing log files
folder_path = r'C:\vijay\research\JVMGC\Sourcecode\Serial_GC\log_file'
process_all_logs_in_folder(folder_path)
