import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


# Augmentation functions
def choose_aug(aug, args):
    imsize = args["imsize"]
    a_end_tf = A.Compose(
        [
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    keep_aspect_resize = A.Compose(
        [
            A.LongestMaxSize(max_size=imsize),
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=cv2.BORDER_CONSTANT, value=0),
        ],
        p=1.0,
    )

    if aug.startswith("aug-02"):
        apply_eq = "EQ" in aug
        apply_bw = "BW" in aug
        keep_aspect = "keep-aspect" in aug
        border = 0
        transform_test = A.Compose(
            [
                A.ToGray(p=1.0) if apply_bw else A.NoOp(),
                A.Equalize(p=1.0) if apply_eq else A.NoOp(),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                A.ToGray(p=1.0) if apply_bw else A.NoOp(),
                A.Equalize(p=1.0) if apply_eq else A.NoOp(),
                A.Posterize(p=0.1),
                A.NoOp() if apply_eq else A.Equalize(p=0.2),
                A.CLAHE(clip_limit=2.0),
                A.OneOf(
                    [
                        A.GaussianBlur(),
                        A.Sharpen(),
                    ],
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Solarize(p=0.2),
                A.OneOf(
                    [
                        A.ColorJitter(),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2
                        ),
                        A.NoOp() if apply_bw else A.ToGray(p=0.5),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.Rotate(limit=360, border_mode=border),
                        A.Perspective(pad_mode=border),
                        A.Affine(
                            scale=(0.5, 0.9),
                            translate_percent=0.1,
                            shear=(-30, 30),
                            rotate=360,
                        ),
                    ],
                    p=0.8,
                ),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                A.CoarseDropout(
                    max_holes=30,
                    max_height=15,
                    max_width=15,
                    min_holes=1,
                    min_height=2,
                    min_width=2,
                ),
                A.RandomSizedCrop(
                    min_max_height=(int(0.5 * imsize), int(0.8 * imsize)),
                    height=imsize,
                    width=imsize,
                    p=0.3,
                ),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]
    else:
        raise ValueError(f"Invalid augmentation value {aug}")

    return tf_test, tf_train

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, split, split_column, transform=None, aug=None, dataset_type='F1'):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.split_column = split_column
        self.transform = transform
        self.aug = aug
        self.dataset_type = dataset_type

        self.data_frame = self.data_frame[self.data_frame[self.split_column] == self.split]

        if self.dataset_type == 'F1':
            self.data_frame['label'] = pd.Categorical(self.data_frame['taxon']).codes
            self.num_classes = len(pd.Categorical(self.data_frame['taxon']).categories)
        elif self.dataset_type == 'F2':
            pass
        else:
            raise ValueError("dataset_type must be either 'F1' or 'F2'")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.dataset_type == 'F1':
            taxon = self.data_frame.iloc[idx, 0]
            image_name = self.data_frame.iloc[idx, 2]
            img_path = os.path.join(self.root_dir, taxon, image_name)
            label = self.data_frame.iloc[idx, -1]
        else:  # F2
            individual = self.data_frame.iloc[idx, 0]
            taxon = self.data_frame.iloc[idx, 1]
            image_name = self.data_frame.iloc[idx, 6]
            img_path = os.path.join(self.root_dir, individual, image_name)
            label = pd.Categorical(self.data_frame['taxon']).codes[idx]

        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        if self.aug:
            augmented = self.aug(image=image)
            image = augmented['image']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Augmentation setup
IMSIZE = 224
args = {"imsize": IMSIZE}
tf_test, tf_train = choose_aug("aug-02", args)

# Constants
DATA_DIR2 = '/scratch/project_2010727/data/IDA/Images'
CSV_FILE2 = '/projappl/project_2010727/DA/Processed-data/Fin2_5splits_taxon.csv'
DATA_DIR1 = '/scratch/project_2010727/data/Detect dataset/Cropped images'
CSV_FILE1 = '/projappl/project_2010727/DA/Processed-data/Fin1_5splits_taxon.csv'
SPLIT_COLUMN = '0'

def load_data(data_folder, batch_size, train, num_workers=40, **kwargs):
    # This function now uses the CustomDataset class
    dataset = CustomDataset(
        csv_file=CSV_FILE2 if 'IDA' in data_folder else CSV_FILE1,
        root_dir=data_folder,
        split='train' if train else 'val',
        dataset_type='F2' if 'IDA' in data_folder else 'F1',
        split_column=SPLIT_COLUMN,
        transform=tf_train if train else tf_test
    )
    
    data_loader = get_data_loader(dataset, batch_size=batch_size, 
                                  shuffle=True if train else False, 
                                  num_workers=num_workers, **kwargs, 
                                  drop_last=True if train else False)
    
    n_class = dataset.num_classes if hasattr(dataset, 'num_classes') else len(dataset.data_frame['taxon'].unique())
    return data_loader, n_class

def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=40, infinite_data_loader=False, **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=40, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0  # Always return 0

# Create datasets
source_loader, source_n_class = load_data(DATA_DIR2, batch_size=32, train=True)
val_loader, _ = load_data(DATA_DIR2, batch_size=32, train=False)
target_train_loader, target_n_class = load_data(DATA_DIR1, batch_size=32, train=True)
target_test_loader, _ = load_data(DATA_DIR1, batch_size=32, train=False)




