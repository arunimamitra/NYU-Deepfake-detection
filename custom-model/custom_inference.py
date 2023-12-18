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
    parser.add_argument('--model_weights', type=str, default='Demo', help='model-weights path')
    args = parser.parse_args()

    checkpoint_filepath=args.model_weights

    #DATA PREPARATION
    test_real_dir = '/scratch/ag8733/CV/real_vs_fake/real-vs-fake/test/real/'
    test_real_path = os.listdir(test_real_dir)
    test_fake_dir = '/scratch/ag8733/CV/real_vs_fake/real-vs-fake/test/fake/'
    test_fake_path = os.listdir(test_fake_dir)
    test_real_df = pd.DataFrame({'image_path': test_real_dir + test_real_path[i], 'label': 1} for i in range(0, 10000))
    test_fake_df = pd.DataFrame({'image_path': test_fake_dir + test_fake_path[i], 'label': 0} for i in range(0, 10000))
    test_df = pd.concat([test_real_df, test_fake_df], ignore_index=True)
    
    

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
    
    test_label = test_df['label']
    test_features = test_df['image_path']
    test_dataset = ImageDataset(test_label,test_features,transform=image_transforms['baseline'])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    print(f"Successuflly loaded test loader: {len(test_loader.dataset)}\n")


    model=CustomNet()
    model=model.to(device)
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001,weight_decay=1e-5,eps=1e-7)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    print(f"Using {device}")
    print("Testing started")
    with torch.no_grad():
        model.eval()
        test_correct=0
        tp, tn, fp, fn = 0, 0, 0, 0  # Initialize TP, TN, FP, FN counters

        for image, target in tqdm(test_loader):
            image, target = image.to(device), target.to(device)
            target = target.unsqueeze(1)

            outputs = torch.sigmoid(model(image))
            predicted_labels = (outputs > 0.5).float()

            test_correct += torch.sum(predicted_labels == target)

            # Calculate TP, TN, FP, FN
            tp += torch.sum((predicted_labels == 1) & (target == 1))
            tn += torch.sum((predicted_labels == 0) & (target == 0))
            fp += torch.sum((predicted_labels == 1) & (target == 0))
            fn += torch.sum((predicted_labels == 0) & (target == 1))

        test_accuracy = test_correct / len(test_loader.dataset)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'Test accuracy: {test_accuracy}')
    print(f'True Positive (TP): {tp}, True Negative (TN): {tn}')
    print(f'False Positive (FP): {fp}, False Negative (FN): {fn}')
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')




if __name__=="__main__":
    main()
