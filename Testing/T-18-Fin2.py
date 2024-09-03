import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch import nn
import cv2
import numpy as np
import torchmetrics

class Dataset(Dataset):
    def __init__(self, csv_file, root_dir, split, split_column, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.split_column = split_column
        self.transform = transform
        self.data_frame = self.data_frame[self.data_frame[self.split_column] == self.split]
        self.num_classes = len(pd.Categorical(self.data_frame['taxon']).categories)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        individual = self.data_frame.iloc[idx, 0]
        taxon = self.data_frame.iloc[idx, 1]
        image_name = self.data_frame.iloc[idx, 6]
        img_path = os.path.join(self.root_dir, individual, image_name)
        
        # Load image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply Value channel processing
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to 3-channel grayscale
        image = np.stack((image,)*3, axis=-1)

        if self.transform:
            image = self.transform(image)

        label = pd.Categorical(self.data_frame['taxon']).codes[idx]
        return image, torch.tensor(label, dtype=torch.long)

class TestModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.test_acc(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc, prog_bar=True)
        return {'test_loss': loss, 'test_acc': self.test_acc}

def main():
    # Set paths
    data_dir = '/scratch/project_2010727/data/IDA/Images'
    csv_file = '/projappl/project_2010727/DA/Processed-data/Fin2_5splits_taxon.csv'
    model_path = '/projappl/project_2010727/Transfer Learning/Saved_Models/ResNet18_VC25_lr0.0001_e4.pth'

    # Set parameters
    num_classes = 25
    batch_size = 264
    num_workers = 40

    # Define transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create test dataset and dataloader
    test_dataset = Dataset(csv_file=csv_file, root_dir=data_dir, split='test', split_column='0', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize the model
    model = TestModel(num_classes=num_classes)

    # Load the saved weights
    state_dict = torch.load(model_path)
    new_state_dict = {"model." + k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict )

    # Initialize trainer
    trainer = Trainer()

    # Test the model
    test_result = trainer.test(model, test_dataloader)

    print(f"Test results: {test_result}")

if __name__ == "__main__":
    main()