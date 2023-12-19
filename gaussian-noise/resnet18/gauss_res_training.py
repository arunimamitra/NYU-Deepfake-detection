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

class DeepfakeDataset(Dataset):
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

def load_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            images.append(img_path)
            labels.append(label)
    return images, labels


def training_loop(model, training_loader, validation_loader, criterion, optimizer, scheduler, checkpoint_folder,epochs=1,start_epoch=0):
    '''Training loop for train and eval modes'''
    best_val_accuracy=0.0
    for epoch in range(start_epoch, start_epoch+epochs):
        model.train()
        train_accuracy = 0
        train_correct=0
        train_loss = 0
        for i,(image, target) in enumerate(tqdm(training_loader,unit='batch')):
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
                print(f"Epoch {epoch+1}/{epochs+start_epoch}, Batch {i} Training Loss: {loss.item()}")

        with torch.no_grad():
            model.eval()
            valid_loss = 0
            val_correct = 0
            for val_image, val_target in validation_loader:
                val_image = val_image.to(device)
                val_target = val_target.to(device)
                val_target = val_target.unsqueeze(1)
                val_outputs = torch.sigmoid(model(val_image))
                val_loss = criterion(val_outputs.float(), val_target.float())
                valid_loss += val_loss.item()
                val_correct += torch.sum((val_outputs > 0.5) == val_target)

        avg_train_loss=train_loss/len(training_loader.dataset)
        avg_val_loss=val_loss /len(validation_loader.dataset)
        train_accuracy=train_correct /len(training_loader.dataset)
        val_accuracy= val_correct /len(validation_loader.dataset)
        print(f'Epoch: {epoch+1}/{epoch+ start_epoch} Average Train loss: { avg_train_loss} Train accuracy: {train_accuracy} Val loss: {  avg_val_loss} Average Val accuracy: {val_accuracy}')
        scheduler.step()

        checkpoint_name=f"checkpoint_epoch{start_epoch+epoch+1}.pth"
        checkpoint_path=os.path.join(checkpoint_folder,checkpoint_name)
        torch.save({'epoch': epoch+start_epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, checkpoint_path)
        if best_val_accuracy<val_accuracy:
            best_val_accuracy=val_accuracy
            checkpoint_path=os.path.join(checkpoint_folder,f"RGbest_model_{start_epoch+epoch+1}.pth")
            torch.save({'epoch': epoch+start_epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, checkpoint_path)


def main():
    ROOT_DIRECTORY = "/scratch/am13018/"

    # Load deepfake and real images
    TRAIN_DATA_PATH=ROOT_DIRECTORY+"/real_vs_fake/real-vs-fake/train"
    VAL_DATA_PATH=ROOT_DIRECTORY+"/real_vs_fake/real-vs-fake//valid"

    train_real_dir = os.path.join(TRAIN_DATA_PATH,'real/')
    train_fake_dir = os.path.join(TRAIN_DATA_PATH,'fake/')
    val_real_dir = os.path.join(VAL_DATA_PATH,'real/')
    val_fake_dir = os.path.join(VAL_DATA_PATH,'fake/')
    train_real_path=os.listdir(train_real_dir)
    train_fake_path=os.listdir(train_fake_dir)
    val_real_path=os.listdir(val_real_dir)
    val_fake_path=os.listdir(val_fake_dir)
   
    num_epochs = 30
    train_size=50000
    val_size=10000
    train_bs=64
    val_bs=32
    lr=0.00001
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5,eps=1e-7)

    # Create custom datasets and data loaders
    train_real_df = pd.DataFrame({'image_path': train_real_dir + train_real_path[i], 'label': 1} for i in range(train_size))
    train_fake_df = pd.DataFrame({'image_path': train_fake_dir + train_fake_path[i], 'label': 0} for i in range(train_size))
    train_df= shuffle(pd.concat([train_real_df, train_fake_df], ignore_index=True)).reset_index(drop=True)

    val_real_df = pd.DataFrame({'image_path': val_real_dir + val_real_path[i], 'label': 1} for i in range(val_size))
    val_fake_df = pd.DataFrame({'image_path': val_fake_dir + val_fake_path[i], 'label': 0} for i in range(val_size))
    val_df= shuffle(pd.concat([val_real_df, val_fake_df], ignore_index=True)).reset_index(drop=True)

    #TRANSFORMATIONS

    image_transforms = {'train_transform':
                        A.Compose([A.Resize(224, 224),
                        A.GaussNoise(var_limit=(1, 2500), mean=2, always_apply=False, p=0.8),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                        ToTensorV2()]),
                        'validation_transform':
                        A.Compose([A.Resize(224, 224),
                        A.GaussNoise(var_limit=(1, 2500), mean=2, always_apply=False, p=0.8),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                        ToTensorV2()]),
                    }
    train_label = train_df['label']
    train_features = train_df['image_path']
    val_label = val_df['label']
    val_features = val_df['image_path']

    train_dataset = DeepfakeDataset(train_label, train_features, transform=image_transforms['train_transform'])
    val_dataset = DeepfakeDataset(val_label, val_features, transform=image_transforms['validation_transform'])
    #visual_train_dataset =  ImageDataset(train_label, train_features, transform=image_transforms['visualization_transform'])

    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=True)
    checkpoint_path = os.path.join('/scratch/am13018/gauss_checkpoints/realresnet/', "")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    

    existing_checkpoints = [f for f in os.listdir(checkpoint_path) if
                            f.startswith('checkpoint_epoch') and f.endswith('.pth')]
    if existing_checkpoints:
        last_checkpoint = max(existing_checkpoints)
        checkpoint_filepath = os.path.join(checkpoint_path, last_checkpoint)
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.to(device)
        start_epoch = checkpoint['epoch'] + 1
        print(f"starting training from epoch-{start_epoch}")
    else:
        start_epoch = 0
        model.to(device)
    print(f"Device:{device}")
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.97)
    criterion= torch.nn.BCELoss()
    print(f"*****TRAINING STARTING*****")
    training_loop(model,train_loader,val_loader, criterion, optimizer,scheduler,checkpoint_path,epochs=num_epochs,start_epoch=0)
    print(f"*****TRAINING COMPLETE*****")




if __name__=="__main__":
    main()
