import albumentations
from torch.utils.data import DataLoader
from wtfml.data_loaders.image import ClassificationLoader
from config.config import TRAIN_BS, VALID_BS, MEAN, STD

def train_albumentation():
    augmentation = albumentations.Compose(
        [
            albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )
    return augmentation


def valid_albumentation():
    augmentation = albumentations.Compose(
        [
            albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True)
        ]
    )
    return augmentation


def train_dataloader(images, targets):
    dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=train_albumentation()
    )
    data_loader = DataLoader(
        dataset=dataset, batch_size=TRAIN_BS, shuffle=True, num_workers=10
    )

    return data_loader


def valid_dataloader(images, targets):
    dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=valid_albumentation()
    )
    data_loader = DataLoader(
        dataset=dataset, batch_size=VALID_BS, shuffle=False, num_workers=10
    )

    return data_loader


def test_dataloader(images, targets):
    dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=valid_albumentation()
    )
    data_loader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=0
    )

    return data_loader