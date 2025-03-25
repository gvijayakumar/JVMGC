import os
import csv

# Directory containing log files
log_dir = r"C:\vijay\research\JVMGC\Sourcecode\Serial_GC\log_file"
output_file = "extracted_data.csv"

# Fields for the CSV file
fields = [
    'Field1', 'Field2', 'Field3', 'Field4', 'Field5', 'Field6', 'Field7', 'Field8', 
    'Field9', 'Field10', 'Field11', 'Field12', 'Field13', 'Field14', 'Field15'
]

# Function to parse a log file
def parse_log_file(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            if ';' in line:
                parts = line.strip().split(';')
                data.append(parts)
    return data

# Create a CSV file and write the header
with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

    # Iterate over each file in the directory
    for filename in os.listdir(log_dir):
        if filename.endswith("SGC.log"):
            filepath = os.path.join(log_dir, filename)
            file_data = parse_log_file(filepath)

            # Write the extracted data to the CSV file
            for row in file_data:
                csvwriter.writerow(row)

print(f"Data extraction complete. Extracted data saved to {output_file}")
