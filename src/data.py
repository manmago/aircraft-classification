"""Reusable data pipeline for the aircraft classifier.

This is the logic from notebooks/02_data_pipeline.ipynb, consolidated so any
notebook or script can get ready-to-use DataLoaders with a single call:

    from src.data import get_dataloaders
    train_loader, val_loader, test_loader, classes = get_dataloaders('../data')
"""
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import FGVCAircraft

# --- configuration --------------------------------------------------------
MY_CLASSES = ['Boeing 737', 'A320', 'A340', 'Boeing 777', 'A380']
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BANNER_PX = 20  # FGVC images have a 20px copyright banner along the bottom


def remove_banner(img):
    """Crop off the bottom 20px FGVC copyright banner (their docs require this)."""
    w, h = img.size
    return img.crop((0, 0, w, h - BANNER_PX))


# TRAIN: remove banner -> random crop + flip (augmentation) -> tensor -> normalize.
train_tf = transforms.Compose([
    transforms.Lambda(remove_banner),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# VAL / TEST: deterministic resize + center crop -> tensor -> normalize.
eval_tf = transforms.Compose([
    transforms.Lambda(remove_banner),
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FilteredAircraft(Dataset):
    """FGVCAircraft restricted to `class_names`, labels remapped to 0..n-1."""

    def __init__(self, base_ds, class_names):
        self.base = base_ds
        self.classes = class_names
        keep_ids = [base_ds.classes.index(name) for name in class_names]
        self.old_to_new = {old: new for new, old in enumerate(keep_ids)}
        self.indices = [i for i, lab in enumerate(base_ds._labels)
                        if lab in self.old_to_new]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, old_label = self.base[self.indices[i]]
        return img, self.old_to_new[old_label]


def _make_split(data_dir, split, tf):
    base = FGVCAircraft(root=str(data_dir), split=split,
                        annotation_level='family', download=False)
    base.transform = tf
    return FilteredAircraft(base, MY_CLASSES)


def get_dataloaders(data_dir='data', batch_size=32):
    """Build the filtered train/val/test DataLoaders.

    Returns (train_loader, val_loader, test_loader, classes).
    `data_dir` is where FGVC was downloaded (e.g. '../data' from a notebook).
    """
    data_dir = Path(data_dir).resolve()
    train_ds = _make_split(data_dir, 'train', train_tf)
    val_ds = _make_split(data_dir, 'val', eval_tf)
    test_ds = _make_split(data_dir, 'test', eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader, MY_CLASSES
