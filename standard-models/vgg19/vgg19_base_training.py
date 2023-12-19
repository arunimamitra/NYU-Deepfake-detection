import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

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
            checkpoint_path=os.path.join(checkpoint_folder,f"best_model_{start_epoch+epoch+1}.pth")
            torch.save({'epoch': epoch+start_epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, checkpoint_path)



def main():
    ROOT_DIRECTORY = "/scratch/am13018/"

    # Load deepfake and real images
    train_deepfake_dir = "real_vs_fake/real-vs-fake/train/fake/"
    train_real_dir = "real_vs_fake/real-vs-fake/train/real"

    valid_deepfake_dir = "real_vs_fake/real-vs-fake/valid/fake/"
    valid_real_dir = "real_vs_fake/real-vs-fake/valid/real/"

    deepfake_images, deepfake_labels = load_images(train_deepfake_dir, label=1)
    real_images, real_labels = load_images(train_real_dir, label=0)

    valid_deepfake_images, valid_deepfake_labels = load_images(valid_deepfake_dir, label=1)
    valid_real_images, valid_real_labels = load_images(valid_real_dir, label=0)

    # Combine deepfake and real images
    all_train_images = deepfake_images + real_images
    all_train_labels = deepfake_labels + real_labels

    # Encode labels
    label_encoder = LabelEncoder()
    all_trainlabels_encoded = label_encoder.fit_transform(all_train_labels)

    all_valid_images = valid_deepfake_images + valid_real_images
    all_valid_labels = valid_deepfake_labels + valid_real_labels
    label_encoder = LabelEncoder()
    all_validlabels_encoded = label_encoder.fit_transform(all_valid_labels)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom datasets and data loaders
    train_dataset = DeepfakeDataset(all_train_images, all_trainlabels_encoded, transform=transform)
    valid_dataset = DeepfakeDataset(all_valid_images, all_validlabels_encoded, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    checkpoint_path = os.path.join('/scratch/am13018/Checkpoint_VGG19Baseline/', "")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    num_epochs = 30
    train_size = 50000
    val_size = 10000
    train_bs = 64
    val_bs = 32
    lr = 0.00001
    model = models.vgg19()
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5, eps=1e-7)

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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    criterion = torch.nn.BCELoss()
    print("*****TRAINING STARTING*****")
    training_loop(model,train_loader,valid_loader, criterion, optimizer,scheduler,checkpoint_path,epochs=num_epochs,start_epoch=0)
    print("*****TRAINING COMPLETE*****")




if __name__=="__main__":
    main()
