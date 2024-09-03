import argparse
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
import numpy as np
import random
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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

class Target(Dataset):
    def __init__(self, csv_file, root_dir, split, split_column, transform=None, use_subset=False):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.split_column = split_column
        self.transform = transform
        self.use_subset = use_subset

        self.data_frame = self.data_frame[self.data_frame[self.split_column] == split]
        self.data_frame['label'] = pd.Categorical(self.data_frame['taxon']).codes
        self.num_classes = len(pd.Categorical(self.data_frame['taxon']).categories)

        if self.use_subset and self.split == 'train':
            subset_size = int(1 * len(self.data_frame))
            self.data_frame = self.data_frame.sample(n=subset_size, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        taxon = self.data_frame.iloc[idx, 0]
        image_name = self.data_frame.iloc[idx, 2]
        img_path = os.path.join(self.root_dir, taxon, image_name)
        
        # Load image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply Value channel processing
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to 3-channel grayscale
        image = np.stack((image,)*3, axis=-1)

        if self.transform:
            image = self.transform(image)

        label = self.data_frame.iloc[idx, -1]
        return image, torch.tensor(label, dtype=torch.long)

class Train(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 25)
        
        # Load the pre-trained weights
        state_dict = torch.load('/projappl/project_2010727/Transfer Learning/Saved_Models/ResNet18_VC25_lr0.0001_e4.pth')
        self.model.load_state_dict(state_dict, strict=True)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
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
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

def main(experiment_name, learning_rate):
    # Set paths
    data_dir = '/scratch/project_2010727/data/Detect dataset/Cropped images'
    csv_file = '/projappl/project_2010727/DA/Processed-data/Fin1_5splits_taxon.csv'

    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    split_column = '0'

    # Create datasets
    train_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='train', split_column=split_column, transform=transform, use_subset=True)
    val_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='val', split_column=split_column, transform=transform)
    test_dataset = Target(csv_file=csv_file, root_dir=data_dir, split='test', split_column=split_column, transform=transform)

    # Set up data loaders
    num_workers = 40
    batch_size = 64
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Set up TensorBoard logger with the experiment name
    logger = TensorBoardLogger("tb_logs", name=experiment_name)

    # Set up ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
    )

    # Create model and trainer
    model = Train(num_classes=train_dataset.num_classes, learning_rate=learning_rate)
    trainer = pl.Trainer(
        max_epochs=50,
        deterministic=True,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,  # Log every 10 steps
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Test the model
    test_results = trainer.test(model, dataloaders=test_dataloader)
    print(test_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Resnet152 model')
    parser.add_argument('--experiment_name', type=str, default='my_model', help='Name for the TensorBoard experiment')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the model')
    args = parser.parse_args()
    main(args.experiment_name, args.learning_rate)