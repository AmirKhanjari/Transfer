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
                # Possible equalization
                A.ToGray(p=1.0) if apply_bw else A.NoOp(),
                A.Equalize(p=1.0) if apply_eq else A.NoOp(),
                # Slow pixel tf
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
                # Colors
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
                # Slow geometrical tf
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
    def __init__(self, csv_file, root_dir, split, split_column, transform=None, aug=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.split_column = split_column
        self.transform = transform
        self.aug = aug

        self.data_frame = self.data_frame[self.data_frame[self.split_column] == split]
        self.data_frame['label'] = pd.Categorical(self.data_frame['taxon']).codes
        self.num_classes = len(pd.Categorical(self.data_frame['taxon']).categories)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        taxon = self.data_frame.iloc[idx, 0]
        image_name = self.data_frame.iloc[idx, 2]
        img_path = os.path.join(self.root_dir, taxon, image_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.aug:
            augmented = self.aug(image=image)
            image = augmented['image']

        if self.transform:
            image = self.transform(image)

        label = self.data_frame.iloc[idx, -1]
        return image, torch.tensor(label, dtype=torch.long)
    
class Resnet18_unfreezingmodel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        
        self.model = models.resnet18()

        state_dict = torch.load('resnet.pth')
        self.model.load_state_dict(state_dict, strict=False)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        self.freeze()
        
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

    def unfreeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.train_acc(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.val_acc(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.test_acc(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
    



def train_unfreeze(model, train_dataloader, val_dataloader, test_dataloader, max_epochs_per_stage=25):
    results = []
    
    stages = [
        ("Stage 1: Training final layer", None),
        ("Stage 2: Unfreezing layer4", model.model.layer4),
        ("Stage 3: Unfreezing layer3", model.model.layer3),
        ("Stage 4: Unfreezing layer2", model.model.layer2),
        ("Stage 5: Unfreezing layer1", model.model.layer1),
        ("Stage 6: Unfreezing conv1", model.model.conv1),
    ]

    for stage_name, layer_to_unfreeze in stages:
        print(stage_name)
        
        if layer_to_unfreeze:
            model.unfreeze_layer(layer_to_unfreeze)
        
        trainer = pl.Trainer(max_epochs=max_epochs_per_stage)
        trainer.fit(model, train_dataloader, val_dataloader)
        
        test_result = trainer.test(model, test_dataloader)
        results.append((stage_name, test_result[0]['test_acc']))
        print(f"Test Accuracy after {stage_name}: {test_result[0]['test_acc']:.4f}")
        

    return model, results

# Data    
data_dir = '/scratch/project_2010727/data/Detect dataset/Cropped images'
csv_file = '/scratch/project_2010727/01_finbenthic1-1_processed_5splits_taxon.csv'

args = {"imsize": 224}

tf_test, tf_train = choose_aug("aug-02", args)

split_column = '0'

train_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='train', split_column=split_column, transform=tf_train)
val_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='val', split_column=split_column, transform=tf_test)
test_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='test', split_column=split_column, transform=tf_test)

num_workers = 30 
batch_size = 264

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_classes = train_dataset.num_classes 
model = Resnet18_unfreezingmodel(num_classes)
final_model, stage_results = train_unfreeze(model, train_dataloader, val_dataloader, test_dataloader)

for stage, acc in stage_results:
    print(f"{stage} - Test Accuracy: {acc:.4f}")
