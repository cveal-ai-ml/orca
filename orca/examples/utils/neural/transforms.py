"""
Author: MINDFUL
Purpose: Data augmentations
"""


import albumentations as album

from albumentations.pytorch import ToTensorV2


def load_data_transforms(choice, interpolate, data_shape):
    """
    Load dataset image transformations / augmentations

    Parameters:
    - choice (int): signifier for augmentation strategy
    - interpolate (int): flag for enabling image resizing
    - data_shape (list[int]): data observation shape (channels, height, width)

    Returns:
    - (dict[str, any]): dataset augmentations
    """

    num_channels, height, width = data_shape

    results = {}
    transforms = []

    if choice == 0:

        if interpolate:
            transforms.append(album.Resize(height, width))

        other = [album.Normalize(),
                 ToTensorV2()]

        transforms = transforms + other

        results["train"] = album.Compose(transforms)
        results["valid"] = album.Compose(transforms)

    elif choice == 1:

        if interpolate:
            operation = album.Resize(height, width)
            transforms.append(operation)

        train = [album.ShiftScaleRotate(p=0.5),
                 album.RGBShift(p=0.5),
                 album.RandomBrightnessContrast(p=0.5),
                 album.augmentations.transforms.GaussNoise(p=0.5),
                 album.Normalize(),
                 ToTensorV2()]

        valid = [album.Normalize(),
                 ToTensorV2()]

        train_transforms = transforms + train
        valid_transforms = transforms + valid

        results["train"] = album.Compose(train_transforms)
        results["valid"] = album.Compose(valid_transforms)

    elif choice == 2:

        if interpolate:
            operation = album.Resize(height, width)
            transforms.append(operation)

        test = [album.Normalize(),
                ToTensorV2()]

        transforms = transforms + test

        results["train"] = None
        results["valid"] = album.Compose(transforms)

    else:

        raise NotImplementedError

    return results
