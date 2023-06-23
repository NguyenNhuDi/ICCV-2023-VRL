from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    Orientationd,
    Spacingd,
    EnsureTyped,
    RandAffined
)

def get_train_augmentations(
        rotate_p:float = 0.5,
        patch_shape=(64, 64, 64),
    ):
    """
    This is where we define our training augmentations.
    """
    transforms = [
        LoadImaged(keys=['img', 'seg']),
        EnsureChannelFirstd(keys=['img', 'seg']),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        Spacingd(keys=["img", "seg"], pixdim=(1, 1, 1),mode=("bilinear", "nearest")),
        ScaleIntensityd(keys='img'),
        EnsureTyped(keys=["img", "seg"], track_meta=False),
        RandCropByPosNegLabeld(keys=['img', 'seg'], label_key='seg', spatial_size=patch_shape, num_samples=10),
        RandAffined(keys=['img', 'seg'])
    ]

    return Compose(transforms)

def get_validation_transformations(patch_shape=(64, 64, 64)):
    """
    This is where we define our validation augmentations. Currently just normalizes.
    """
    transforms = [
        LoadImaged(keys=['img', 'seg']),
        EnsureChannelFirstd(keys=['img', 'seg']),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        Spacingd(keys=["img", "seg"], pixdim=(1, 1, 1),mode=("bilinear", "nearest")),
        ScaleIntensityd(keys='img'),
        EnsureTyped(keys=["img", "seg"], track_meta=False),

        RandCropByPosNegLabeld(keys=['img', 'seg'], label_key='seg', spatial_size=patch_shape, num_samples=10),
    ]

    return Compose(transforms)