import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json

class RHDDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        with open(annotations_file) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations[idx]["image"])
        image = read_image(img_path)
        joints = torch.tensor(self.annotations[idx]["joints"])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            joints = self.target_transform(joints)
        return image, joints

from torch.utils.data import DataLoader

def create_data_loaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Specify the paths to your data.
train_img_dir = "../data/RHD_published_v2/training/color"
val_img_dir = "../data/RHD_published_v2/evaluation/color"
train_annotations = "../data/RHD_published_v2/training/anno_training.json"
val_annotations = "../data/RHD_published_v2/evaluation/anno_evaluation.json"

# Create your datasets.
train_dataset = RHDDataset(train_img_dir, train_annotations)
val_dataset = RHDDataset(val_img_dir, val_annotations)

# Create your data loaders.
train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

