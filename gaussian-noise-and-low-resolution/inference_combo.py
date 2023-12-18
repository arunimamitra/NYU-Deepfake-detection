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
import argparse

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

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--model_weights', type=str, default='Demo', help='model-weights path')
    parser.add_argument('--model_type', type=str, default='vgg19', help='model-type')
    args = parser.parse_args()

    checkpoint_filepath=args.model_weights
    model_type=args.model_type

    test_real_dir = '/scratch/ag8733/CV/real_vs_fake/real-vs-fake/test/real/'
    test_real_path = os.listdir(test_real_dir)
    test_fake_dir = '/scratch/ag8733/CV/real_vs_fake/real-vs-fake/test/fake/'
    test_fake_path = os.listdir(test_fake_dir)
    test_real_df = pd.DataFrame({'image_path': test_real_dir + test_real_path[i], 'label': 1} for i in range(0, 10000))
    test_fake_df = pd.DataFrame({'image_path': test_fake_dir + test_fake_path[i], 'label': 0} for i in range(0, 10000))
    test_df = pd.concat([test_real_df, test_fake_df], ignore_index=True)
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
    test_dataset = ImageDataset(test_label,test_features,transform=image_transforms['combination'])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    if model_type=="vgg19":
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
        print("Choosing vgg19")
    elif model_type=="resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        print("Choosing resnet18")
    elif model_type=="alexnet":
        model=models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)# for alexnet
        print("Choose alexnet")
    else:
        print("Wrong model")
        exit(-1)


    if model_type=="vgg19":
        checkpoint= torch.load(checkpoint_filepath)
        state_dict=checkpoint['model_state_dict']
        remove_prefix = 'module.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
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