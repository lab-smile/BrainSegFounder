import dicom2nifti
import os 
import re
from pathlib import Path
import datetime

from typing import List

dicom2nifti.settings.disable_validate_slice_increment()

## LIDC Data: 
# This is going to be so scuffed LMAO
def count_files(directory: Path, subdirectory: Path) -> int:
    num_files = 0
    for _, _, files in os.walk(directory / subdirectory):
        num_files += len(files)
    return num_files

def convert_to_nifti(directory: Path) -> None:
    DATA_DIRECTORY = Path('./Image_LIDC')
    print(f'Converting {directory}')
    dicom2nifti.convert_directory(directory, DATA_DIRECTORY)
    # This ouputs a seemingly random filename, so we need to search for it and renane
    # I'm sure there's a better way to do this, but this is quick and dirty and it works. 
    file_int =  str(directory).split('/')[1].split('-')[2]  # It'd be nice to make sure this is an int but that would strip 0s which are necessary 
    fname = 'LIDC-IDRI-' + file_int + '_0.nii.gz'
    old_file = [file for file in os.listdir(DATA_DIRECTORY) if \
                (file.endswith('.nii.gz') and not file.startswith('LIDC'))]
    if len(old_file) == 0:
        print('COULD NOT CONVERT, NO NIFTI GENERATED, SKIPPING')
        return
    assert len(old_file) == 1  # only one image SHOULD match that description
    old_file = old_file[0]
    os.rename(DATA_DIRECTORY/old_file, DATA_DIRECTORY/fname)
    
    
def convert_LIDC() -> None:
    data_directory = Path('./Image_LIDC')
    for folder in data_directory.iterdir():
        if re.search(r'\d+$', str(folder)) is not None:
            if len(os.listdir(folder)) == 1:
                correct_folder = os.listdir(folder)[0]
            elif len(os.listdir(folder)) == 2:
                folder_1, folder_2 = os.listdir(folder)
                f1_files, f2_files = [count_files(folder, f) for f in os.listdir(folder)]

                correct_folder = folder_1 if f1_files > f2_files else folder_2
            else:
                folder_1, folder_2, folder_3 = os.listdir(folder)
                f1_files, f2_files, f3_files = [count_files(folder, f) for f in os.listdir(folder)]
                num_files_list = [f1_files, f2_files, f3_files]
                index = num_files_list.index(max(num_files_list))
                correct_folder = (os.listdir(folder)[index])
            convert_to_nifti(folder/correct_folder)          

# HNSCC data, probably not much better
def locate_latest_CT_images(possible_directories:  List[str]) -> List[str]:# list[str]) -> list[str]:
    latest_image = None
    latest_date = None
    for directory in possible_directories:
        image_date = '-'.join(directory.split('-', 3)[:3])
        image_date = datetime.datetime.strptime(image_date, '%m-%d-%Y')
        if latest_image is not None:
            if image_date > latest_date: 
                latest_image, latest_date = directory, image_date
        else: 
            latest_image = directory
            latest_date = image_date  
    return [directory for directory in possible_directories if directory.startswith(latest_image)]


def locate_CT_scan(patient: str, possible_directories: List[str]) -> str:
    for directory in possible_directories: 
        # This needs to happen first: we only want 3.000 images if there are no 2.000 images
        for image in os.listdir(f'images/{patient}/{directory}'):
            if '2.000000' in image:
                if len(possible_directories) > 1:
                    print(f'Found CT scan for {patient} in {directory} with multiple possibilities.')
                return f'images/{patient}/{directory}/{image}'
       
        for image in os.listdir(f'images/{patient}/{directory}'):
            if '3.000000' in image:
                if len(possible_directories) > 1:
                    print(f'Found CT scan for {patient} in {directory} with multiple possibilities.')
                return f'images/{patient}/{directory}/{image}'
        # Sigh    
        for image in os.listdir(f'images/{patient}/{directory}'):
            if '4.000000' in image:
                if len(possible_directories) > 1:
                    print(f'Found CT scan for {patient} in {directory} with multiple possibilities.')
                return f'images/{patient}/{directory}/{image}'
        
    print(f'Could not locate CT scan for {patient} in {possible_directories}. Skipping.')


def locate_HNSCC_image(possible_directories: List[str], patient: str) -> str:
    latest_CT_images = locate_latest_CT_images(possible_directories)
    CT_scan = locate_CT_scan(patient, latest_CT_images)
    return CT_scan
    
def convert_CT_scan(CT_scan: str, file_name: str) -> None:
    if CT_scan is not None:
        dicom2nifti.convert_directory(CT_scan, './images')
        old_file = [file for file in os.listdir('./images') if \
                    (file.endswith('.nii.gz') and not file.startswith('img_'))]
        if len(old_file) == 0:
            print('COULD NOT CONVERT, NO NIFTI GENERATED, SKIPPING')
            return
        old_file = old_file[0]  # Sometimes we have 2 nifti files generated, I hope this picks the right one
        os.rename(f'./images/{old_file}', f'./images/{file_name}')
        print(f'converted to {file_name}')

    
def convert_HNSCC() -> None:
    for patient in os.listdir('images'):
        print(patient, end=' ')
        if patient.endswith('.nii.gz'): continue # Already converted :) 
        patient_id = int(patient.split('-')[-1])  # Why keep leading zeros like you did last time when you can not do that!
        file_name = 'img_' + str(patient_id) + '.nii.gz'  # Yes because images/img_3 makes sure we all know this is from HNSCC and not just an image
        CT_scan = locate_HNSCC_image(os.listdir(f'./images/{patient}'), patient)
        convert_CT_scan(CT_scan, file_name)
    
if __name__ == '__main__':
    # convert_LIDC()
    print('-----')
    convert_HNSCC()

