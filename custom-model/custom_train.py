import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models,transforms

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import  tqdm

#DEVICE CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, image_labels, image_dir, transform=None, target_transform=None):
        self.image_labels = image_labels
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.image_labels)


    def __getitem__(self, index):
        image_path = self.image_dir.iloc[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.image_labels.iloc[index]
        if self.transform:
            image = self.transform(image=image)['image']
        if self.target_transform:
            label = self.target_transform(label=label)
        return image, label

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 28 * 28, 1)
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        return self.fc1(x)


def main():
    
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--checkpoint_path', type=str, default='Demo', help='Path to store checkpoint files')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for training')
    args = parser.parse_args()

    epochs=args.num_epochs
    checkpoint_path = os.path.join('/scratch/ag8733/CV',args.checkpoint_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)


    lr=0.00001
    model=CustomNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5,eps=1e-7)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.97)
    criterion= torch.nn.BCELoss()
    model=model.to(device)

    # Apply Data Parallelism if CUDA count >1
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    
    #DATA PREPARATION
    TRAIN_DATA_PATH="/scratch/ag8733/CV/real_vs_fake/real-vs-fake/train"
    VAL_DATA_PATH="/scratch/ag8733/CV/real_vs_fake/real-vs-fake/valid"
    train_real_dir = os.path.join(TRAIN_DATA_PATH,'real/')
    train_fake_dir = os.path.join(TRAIN_DATA_PATH,'fake/')
    val_real_dir = os.path.join(VAL_DATA_PATH,'real/')
    val_fake_dir = os.path.join(VAL_DATA_PATH,'fake/')
    train_real_path=os.listdir(train_real_dir)
    train_fake_path=os.listdir(train_fake_dir)
    val_real_path=os.listdir(val_real_dir)
    val_fake_path=os.listdir(val_fake_dir)
    train_real_df = pd.DataFrame({'image_path': train_real_dir + train_real_path[i], 'label': 1} for i in range(50000))   # real is class 1
    train_fake_df = pd.DataFrame({'image_path': train_fake_dir + train_fake_path[i], 'label': 0} for i in range(50000))
    train_df= shuffle(pd.concat([train_real_df, train_fake_df], ignore_index=True)).reset_index(drop=True)
    val_real_df = pd.DataFrame({'image_path': val_real_dir + val_real_path[i], 'label': 1} for i in range(5000))
    val_fake_df = pd.DataFrame({'image_path': val_fake_dir + val_fake_path[i], 'label': 0} for i in range(5000))
    val_df= shuffle(pd.concat([val_real_df, val_fake_df], ignore_index=True)).reset_index(drop=True)

    #TRANSFORMATIONS
    image_transforms = {'baseline':
                        A.Compose([A.Resize(224, 224),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                        ToTensorV2()]),

                        'resolution':
                        A.Compose([A.Resize(128, 128),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                        ToTensorV2()]),

                        'combination':
                        A.Compose([A.Resize(128, 128),
                        A.GaussNoise(var_limit=(1, 2500), mean=2, always_apply=False, p=0.8),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                        ToTensorV2()]),

                        'noise':
                        A.Compose([A.Resize(224, 224),
                        A.GaussNoise(var_limit=(1, 2500), mean=2, always_apply=False, p=0.8),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                        ToTensorV2()])}
    

    train_label = train_df['label']
    train_features = train_df['image_path']
    val_label = val_df['label']
    val_features = val_df['image_path']

    train_dataset = ImageDataset(train_label, train_features, transform=image_transforms['baseline'])
    val_dataset = ImageDataset(val_label, val_features, transform=image_transforms['baseline'])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    print(f"Successuflly loaded train and val data loader.Length of train loader: {len(train_loader.dataset)} Val Loader length={len(val_loader.dataset)}\n")

    #MODEL
    existing_checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith('checkpoint_epoch') and f.endswith('.pth')]
    if existing_checkpoints:
        checkpoint_numbers = [f.split('_')[1][5:].split('.')[0] for f in existing_checkpoints]
        print(checkpoint_numbers)
        last_checkpoint_number = max(checkpoint_numbers)
        last_checkpoint = f"checkpoint_epoch{last_checkpoint_number}.pth"
        print(last_checkpoint)
        checkpoint_filepath = os.path.join(checkpoint_path, last_checkpoint)
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.to(device)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Starting training from epoch-{start_epoch}")
    else:
        start_epoch = 0
        model.to(device)
    print(f"Device:{device}")
    print(f"*****TRAINING STARTING*****")

    best_val_accuracy=0.0
    for epoch in range(start_epoch, start_epoch+epochs):
        model.train()
        train_accuracy = 0
        train_correct=0
        train_loss = 0
        for i,(image, target) in enumerate(tqdm(train_loader,unit='batch')):
            image = image.to(device)
            target = target.to(device)
            target = target.unsqueeze(1)
            optimizer.zero_grad()
            outputs = nn.Sigmoid()(model(image))
            loss = criterion(outputs.float(), target.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            #train_accuracy += ((outputs > 0.5) == target).float().mean().item()
            train_correct+=torch.sum((outputs > 0.5) == target)
            if(i%500==0):
                print(f"Epoch {start_epoch+epoch+1}/{epochs+start_epoch}, Batch {i} Training Loss: {loss.item()}")

        with torch.no_grad():
            model.eval()
            valid_loss = 0
            val_correct = 0
            for val_image, val_target in val_loader:
                val_image = val_image.to(device)
                val_target = val_target.to(device)
                val_target = val_target.unsqueeze(1)
                val_outputs = torch.sigmoid(model(val_image))
                val_loss = criterion(val_outputs.float(), val_target.float())
                valid_loss += val_loss.item()
                val_correct += torch.sum((val_outputs > 0.5) == val_target)

        avg_train_loss=train_loss/len(train_loader.dataset)
        avg_val_loss=val_loss /len(val_loader.dataset)
        train_accuracy=train_correct /len(train_loader.dataset)
        val_accuracy= val_correct /len(val_loader.dataset)
        print(f'Epoch: {start_epoch+epoch+1}/{epochs+ start_epoch} Average Train loss: { avg_train_loss} Train accuracy: {train_accuracy} Val loss: {  avg_val_loss} Average Val accuracy: {val_accuracy}')
        scheduler.step()

        checkpoint_name=f"checkpoint_epoch{start_epoch+epoch+1}.pth"
        checkpoint_path=os.path.join(checkpoint_path,checkpoint_name)
        torch.save({'epoch': epoch+start_epoch+1,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, checkpoint_path)
        if best_val_accuracy<val_accuracy:
            best_val_accuracy=val_accuracy
            checkpoint_path=os.path.join(checkpoint_path,f"best_model_{start_epoch+epoch+1}.pth")
            torch.save({'epoch': epoch+start_epoch+1,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, checkpoint_path)

    print(f"*****TRAINING COMPLETE*****")

if __name__=="__main__":
    main()
