import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import models
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def main():
    test_real_dir = '/scratch/dp3635/real_vs_fake/real-vs-fake/test/real/'
    test_real_path = os.listdir(test_real_dir)
    test_fake_dir = '/scratch/dp3635/real_vs_fake/real-vs-fake/test/fake/'
    test_fake_path = os.listdir(test_fake_dir)
    test_real_df = pd.DataFrame({'image_path': test_real_dir + test_real_path[i], 'label': 1} for i in range(0, 10000))
    test_fake_df = pd.DataFrame({'image_path': test_fake_dir + test_fake_path[i], 'label': 0} for i in range(0, 10000))
    test_df = pd.concat([test_real_df, test_fake_df], ignore_index=True)
    
    image_transforms = {'train_transform':
                        A.Compose([A.Resize(224, 224),
                        A.GaussNoise(var_limit=(10.0, 2500.0), mean=2, always_apply=False, p=1),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                        ToTensorV2()]),
                    'validation_transform':
                    A.Compose([A.Resize(128, 128),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                        ToTensorV2()])
                    }

    test_label = test_df['label']
    test_features = test_df['image_path']
    test_dataset = ImageDataset(test_label,
                                 test_features,
                                 transform=image_transforms['train_transform'])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    model = models.alexnet()
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
    checkpoint_filepath='/scratch/dp3635/gauss_checkpoints/vgg/VGbest_model_25.pth'
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001,weight_decay=1e-5,eps=1e-7)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    print(f"Using {device}")
    print("Testing started")
    with torch.no_grad():
        model.eval()
        test_correct=0
        for image,target in tqdm(test_loader):
            image,target = image.to(device),target.to(device)
            target = target.unsqueeze(1)
            outputs = model(image)
            test_correct += torch.sum((outputs > 0.5) == target)

    print(f'Test accuracy:{test_correct /len(test_loader.dataset)}')

if __name__=="__main__":
    main()
