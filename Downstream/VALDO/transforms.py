from monai import transforms

def get_transforms(key: str, roi: list):
    if key == 'train':
        transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.CropForegroundd(
                    keys=["image", "label"], source_key="image", k_divisible=roi
                ),
                transforms.RandSpatialCropd(
                    keys=["image", "label"], roi_size=roi, random_size=False
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.Resized(keys=["image", "label"], spatial_size=roi),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    
    return transform