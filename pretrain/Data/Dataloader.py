from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)
import json
from monai.data.image_reader import NibabelReader


def load_datalist(input_file: str):
    with open(input_file, 'r') as f:
        fold = json.load(f)
    print(fold.keys())
    training_images = fold['fold_0']['training']  # Should be list
    validation_images = fold['fold_0']['validation']
    t1_path = '/red/ruogu.fang/UKB/Brain/20252_T1_NIFTI/T1_unzip'
    t2_path = '/red/ruogu.fang/UKB/Brain/20253_T2_NIFTI/T2_unzip'
    training = {}
    for i, image in enumerate(training_images):
        image_t1 = t1_path + '/' + image + "_20252_2_0/T1_brain_to_MNI.nii.gz"
        image_t2 = t2_path + '/' + image + "_20253_2_0/T2_FLAIR/T2_FLAIR_brain_to_MNI.nii.gz"
        training[i] = {"T1_image": image_t1,
                       "T2_image": image_t2}

    validation = {}
    for i, image in enumerate(validation_images):
        image_t1 = t1_path + '/' + image + "_20252_2_0/T1_brain_to_MNI.nii.gz"
        image_t2 = t2_path + '/' + image + "_20253_2_0/T2_FLAIR/T2_FLAIR_brain_to_MNI.nii.gz"
        validation[i] = {"T1_image": image_t1,
                         "T2_image": image_t2}
    return {'training': training,
            'validation': validation}


def get_swin_dataloaders(args, modality="T1", num_workers = 4):
    if modality == "T1":
        splits = "../jsons/T1_folds.json"
    elif modality == "T1_T2":
        splits = "/blue/ruogu.fang/cox.j/GB_rewrite/jsons/T1_T2_folds.json"
    else:
        raise ValueError("Unsupported modality")

    datalist = load_datalist(splits)
    training_datalist, validation_datalist = datalist['training'], datalist['validation']
    print(f"Training on {len(training_datalist)} {modality} images.")
    print(f"Validation on {len(validation_datalist)} {modality} images.")

    transforms = Compose(
        [
            LoadImaged(keys=["T1_image", "T2_image"], reader=NibabelReader),
            AddChanneld(keys=["T1_image", "T2_image"]),
            Orientationd(keys=["T1_image", "T2_image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["T1_image", "T2_image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPadd(keys=["T1_image", "T2_image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForegroundd(keys=["T1_image", "T2_image"], source_key="T1_image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["T1_image", "T2_image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["T1_image"]),
        ]
    )

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        training_dataset = CacheDataset(data=training_datalist, transform=transforms, cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        training_dataset = SmartCacheDataset(
            data=training_datalist,
            transform=transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using generic dataset")
        training_dataset = Dataset(data=training_datalist, transform=transforms)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=training_dataset, even_divisible=True, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        training_dataset, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True)

    val_ds = Dataset(data=validation_datalist, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

    return train_loader, val_loader


def get_modelsgenesis_dataloaders():
    #TODO
    pass
