import json

# Load the provided JSON file
file_path = 'brats21_folds.json'  # Replace with your file path

with open(file_path, 'r') as file:
    data = json.load(file)

# Removing the first two items from each 'image' list in the 'training' data
for entry in data['training']:
    entry['image'] = entry['image'][2:]

# Save the modified data back to a new JSON file
modified_file_path = 'brats21_folds_2modalities.json'  # Replace with your desired file path
with open(modified_file_path, 'w') as file:
    json.dump(data, file)
