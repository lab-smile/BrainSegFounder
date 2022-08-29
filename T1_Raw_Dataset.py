import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from numpy import expand_dims
import albumentations as A
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
import SimpleITK as sitk


class T1RawDataset(Dataset):
    def __init__(self, annotations_file, img_dir, input_channels, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.to_float = A.ToFloat()
        self.input_channels = input_channels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0], "/T1.nii.gz")

        itk_img = sitk.ReadImage(img_path) 
        img_array = sitk.GetArrayFromImage(itk_img)
        img_array = img_array.transpose(2, 1, 0)

        

        if self.transform:
            img_array = self.transform(image=img_array)["image"]
            img_array = self.to_float(image=img_array)["image"]

        else:
            img_array = self.to_float(image=img_array)["image"]
        
        

        img_array = img_array.transpose(2,0,1)


        return img_array


class T1RawDatasetViTAutoEnc(Dataset):
    def __init__(self, annotations_file, img_dir, input_channels, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.to_float = A.ToFloat()
        self.input_channels = input_channels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        img_array = read_image(img_path).numpy().transpose(1,2,0)

        

        if self.transform:
            img_array = self.transform(image=img_array)["image"]
            img_array = self.to_float(image=img_array)["image"]

        else:
            img_array = self.to_float(image=img_array)["image"]
        
        if (self.input_channels == 1):
            img_array = expand_dims(img_array[:,:,0], axis=2)

        img_array = img_array.transpose(2,0,1)


        return img_array