import json
from copy import deepcopy
from pathlib import Path
from itertools import chain, combinations
from typing import Iterable

# This code assumes you have downloaded the 4-channel folds json from
# https://drive.google.com/file/d/1i-BXYe-wZ8R9Vp3GXoajGyqaJ65Jybg1/view
# and placed it in the jsons subdirectory.


def name_set(modalities: set):
    base_name = 'brats21_folds'
    for modality in modalities:
        base_name += '_' + modality
    base_name += ".json"
    return base_name


def powerset(iterable: Iterable):
    set_ = list(iterable)
    return chain.from_iterable(combinations(set_, num_selections) for num_selections in range(len(set_)+1))


def subset_modalities(folds_: list, modalities: set, all_modalities: set) -> list:
    removable_modalities = all_modalities - modalities
    print(f'{removable_modalities = }')
    m_list = [name + '.nii.gz' for name in removable_modalities]
    print(f'{m_list}')
    subset_folds = []
    for fil_group in folds_:
        image_list = fil_group['image']
        for image in deepcopy(image_list):
            modality = image.split('_')[-1]
            if modality in m_list:
                image_list.remove(image)


        subset_folds.append({
            'fold': fil_group['fold'],
            'image': image_list,
            'label': fil_group['label']
        })
    return subset_folds


if __name__ == "__main__":
    folds_dir = Path("./jsons/brats21_folds.json")
    possible_modalities = {'flair', 't1ce', 't1', 't2'}
    with open(folds_dir, 'r') as json_file:
        folds = json.load(json_file)
    folds = folds['training']  # The data inside the json file is all under "training" (as opposed to test)
    for desired_modalities in powerset(possible_modalities):
        desired_modalities = set(desired_modalities)
        if not (len(desired_modalities) == 0 or len(desired_modalities) == 4):  # Avoid empty set and full subset
            print(desired_modalities)
            f_name = Path('./jsons') / name_set(desired_modalities)
            print(f_name)
            new = subset_modalities(deepcopy(folds), desired_modalities, possible_modalities)
            new_folds = {'training': new}
            with open(f_name, 'w') as f:
                json.dump(new_folds, f)
