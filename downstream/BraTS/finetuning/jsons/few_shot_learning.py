

import json
from random import shuffle

def split_train_test(data, test_ratio=0.2):
    """
    Splits the data into training and testing subsets based on the specified test ratio.
    """
    shuffle(data)  # Ensure the data is randomly shuffled before splitting
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]

def save_json(file_path, data):
    """
    Saves the given data to a JSON file at the specified file path.
    """
    with open(file_path, 'w') as file:
        json.dump({"data": data}, file)

# Load the initial dataset
file_path = './brats21_folds.json'  # Update this path
with open(file_path, 'r') as file:
    data = json.load(file)['training']

# Convert loaded data into a simpler format (image paths and labels only, no folds)
simplified_data = [{"images": entry["image"], "label": entry["label"]} for entry in data]

# Split the entire dataset into a fixed test set and the remaining data for training
fixed_test_set_ratio = 0.2  # 20% of the data for testing
remaining_data, fixed_test_set = split_train_test(simplified_data, test_ratio=fixed_test_set_ratio)


# Save the fixed test set to JSON
fixed_test_set_file_path = './brats21_fewshot_fixed_test_set.json'
save_json(fixed_test_set_file_path, fixed_test_set)

# Calculate the number of samples for each training subset excluding the fixed test set
updated_total_pairs = len(remaining_data)
split_indices = [int(updated_total_pairs * perc / 100) for perc in [10, 20, 40, 60, 80, 100]]

# Create and save updated training subsets
for perc, split_idx in zip([10, 20, 40, 60, 80, 100], split_indices):
    subset_data = remaining_data[:split_idx]
    updated_train_subset_file_path = f'./brats21_fewshot_train_{perc}%.json'
    save_json(updated_train_subset_file_path, subset_data)
