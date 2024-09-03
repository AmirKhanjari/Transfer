import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms, models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import cv2
import torchmetrics
from efficientnet_pytorch import EfficientNet 
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
import argparse
import random

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

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

class Target(Dataset):
    def __init__(self, csv_file, root_dir, split, split_column, transform=None, aug=None, use_subset=False):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.split_column = split_column
        self.transform = transform
        self.aug = aug
        self.use_subset = use_subset

        self.data_frame = self.data_frame[self.data_frame[self.split_column] == split]
        self.data_frame['label'] = pd.Categorical(self.data_frame['taxon']).codes
        self.num_classes = len(pd.Categorical(self.data_frame['taxon']).categories)

        if self.use_subset and self.split == 'train':
            subset_size = int(0.5 * len(self.data_frame))
            self.data_frame = self.data_frame.sample(n=subset_size, random_state=42).reset_index(drop=True)


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        taxon = self.data_frame.iloc[idx, 0]
        image_name = self.data_frame.iloc[idx, 2]
        img_path = os.path.join(self.root_dir, taxon, image_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.aug:
            augmented = self.aug(image=np.array(image))
            image = augmented['image']

        if self.transform:
            image = self.transform(image)

        label = self.data_frame.iloc[idx, -1]
        return image, torch.tensor(label, dtype=torch.long)
    



class Train(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, fine_tune=False, transfer_state_dict=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.fine_tune = fine_tune
      #  self.save_hyperparameters()
        self.model = models.resnet18(pretrained=True)

        
        if fine_tune:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            self.model.load_state_dict(transfer_state_dict, strict=True)


        
        if not fine_tune:
           num_ftrs = self.model.fc.in_features
           self.model.fc = nn.Linear(num_ftrs, 39)
           state_dict = torch.load('/projappl/project_2010727/Transfer Learning/Saved_Models/ResNet18_epoch4.pth')
           self.model.load_state_dict(state_dict, strict=True)

           num_ftrs = self.model.fc.in_features
           self.model.fc = nn.Linear(num_ftrs, num_classes)

           for param in self.model.parameters():
                param.requires_grad = False

           for param in self.model.fc.parameters():
                param.requires_grad = True


        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.train_acc(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.val_acc(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.test_acc(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        if self.fine_tune:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
        return optimizer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Transfer Learning and Fine-Tuning')
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--transfer_learning_rate', type=float, required=True)
parser.add_argument('--transfer_epochs', type=int, required=True)
parser.add_argument('--fine_tuning_learning_rate', type=float, required=True)
parser.add_argument('--fine_tuning_epochs', type=int, required=True)
args = parser.parse_args()

# Data preparation
data_dir = '/scratch/project_2010727/data/Detect dataset/Cropped images'
csv_file = '/projappl/project_2010727/taxonomist/data/processed/finbenthic1-1/01_finbenthic1-1_processed_5splits_taxon.csv'

aug_args = {"imsize": 224}
tf_test, tf_train = choose_aug("aug-02", aug_args)
split_column = '0'

train_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='train', split_column=split_column, transform=tf_train, use_subset=True)
val_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='val', split_column=split_column, transform=tf_test)
test_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='test', split_column=split_column, transform=tf_test)

num_workers = 40
batch_size = 264

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Step 1: Train the last fully connected layer (Transfer Learning)
transfer_model = Train(num_classes=train_dataset.num_classes, learning_rate=args.transfer_learning_rate)
transfer_trainer = pl.Trainer(max_epochs=args.transfer_epochs, logger=pl.loggers.TensorBoardLogger("tb_logs", name=f"{args.experiment_name}_transfer"))
transfer_trainer.fit(transfer_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Step 2: Fine-tuning the entire model
fine_tune_model = Train(num_classes=train_dataset.num_classes, learning_rate=args.fine_tuning_learning_rate, fine_tune=True, transfer_state_dict=transfer_model.model.state_dict())

fine_tune_trainer = pl.Trainer(max_epochs=args.fine_tuning_epochs, logger=pl.loggers.TensorBoardLogger("tb_logs", name=f"{args.experiment_name}_fine_tune"))
fine_tune_trainer.fit(fine_tune_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


test_results = fine_tune_trainer.test(fine_tune_model, dataloaders=test_dataloader)