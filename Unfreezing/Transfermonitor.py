import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
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
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=cv2.BORDER_CONSTANT, value=0),  # Add the 'value' parameter
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

        # Filter the data frame for the specific split
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
    

class TransferLearningModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        
        # Load a pre-trained EfficientNet-B0 model
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = self.model._fc.in_features
        
        # Replace the final fully connected layer with a new one that matches the number of classes in the target domain
        self.model._fc = nn.Linear(num_ftrs, num_classes)
        
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
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def on_train_end(self):
        train_acc = self.train_acc.compute()
        print(f'Train Accuracy: {train_acc:.4f}')
        self.train_acc.reset()
    
    def on_validation_end(self):
        val_acc = self.val_acc.compute()
        print(f'Validation Accuracy: {val_acc:.4f}')
        self.val_acc.reset()
    
    def on_test_end(self):
        test_acc = self.test_acc.compute()
        print(f'Test Accuracy: {test_acc:.4f}')
        self.test_acc.reset()

# Define the dataset paths
data_dir = '/scratch/project_2010727/data/Detect dataset/Cropped images'
csv_file = '/scratch/project_2010727/01_finbenthic1-1_processed_5splits_taxon.csv'

# Set image size
args = {"imsize": 224}

# Choose the augmentation method
tf_test, tf_train = choose_aug("aug-02", args)

# Specify which split column to use (e.g., '0' for the first split column)
split_column = '0'

# Create the datasets for each split
train_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='train', split_column=split_column, transform=tf_train)
val_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='val', split_column=split_column, transform=tf_test)
test_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='test', split_column=split_column, transform=tf_test)

# Create the DataLoaders for each split
num_workers = 30  # Adjusting based on system recommendation
batch_size = 264

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Function to perform gradual unfreezing and fine-tuning
def gradual_unfreezing():
    # Initialize the model
    model = TransferLearningModel(num_classes=train_dataset.num_classes, learning_rate=1e-3)

    # Load the pre-trained weights from the source domain, excluding the final layer
    pretrained_weights = torch.load('efficientnet_b0_f2.pth')
    pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model.model.state_dict() and not k.startswith('_fc')}
    model.model.load_state_dict(pretrained_weights, strict=False)

    # Freeze all layers initially
    for name, param in model.model.named_parameters():
        param.requires_grad = False

    # List of layer groups to unfreeze gradually
    layer_groups = [model.model._fc]  # Start with the final layer

    # Add blocks to the list of layer groups for gradual unfreezing
    stem = [model.model._conv_stem, model.model._bn0]
    blocks = list(model.model._blocks)
    head = [model.model._conv_head, model.model._bn1]

    layer_groups.extend(reversed(stem + blocks + head))

    results = []

    for i, group in enumerate(layer_groups):
        # Unfreeze the current group
        for param in group.parameters():
            param.requires_grad = True

        # Reset the optimizer
        model.configure_optimizers()

        # Initialize a new trainer for each step
        trainer = pl.Trainer(max_epochs=25)  # Adjust the number of epochs as needed

        # Fine-tune the model
        print(f"Unfreezing step {i+1}: {group}")
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Test the model
        test_results = trainer.test(model, dataloaders=test_dataloader)
        test_acc = model.test_acc.compute()
        results.append((f'Step {i+1}', test_acc.item()))
        print(f'Test Accuracy after unfreezing step {i+1}: {test_acc:.4f}')
        model.test_acc.reset()

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Step', 'Test Accuracy'])
    return results_df

# Perform gradual unfreezing
unfreeze_results_df = gradual_unfreezing()

# Display the results as a table
print(unfreeze_results_df)
