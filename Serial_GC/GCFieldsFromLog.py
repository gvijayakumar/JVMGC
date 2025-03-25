import re

# Open the GC log file
with open("C:\\vijay\\research\\JVMGC\\Source code\\Serial_GC\\log_file\\Xms2m_Xmx100m_SGC", "r") as gc_log:
    log_text = gc_log.read()

# Use regular expressions to find all key-value pairs in the log text
key_value_pairs = re.findall(r'(\S+): (\S+)', log_text)

# Print the key-value pairs
for key, value in key_value_pairs:
    print(key, ':', value)
