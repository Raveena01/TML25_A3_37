# utils.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
class TaskDataset:
    pass 

def load_train_dataset(pt_file="Train.pt"):
    data = torch.load(pt_file, map_location="cpu")
    return data.imgs, data.labels


def get_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()  # ONLY this
    ])

class WrappedDataset(Dataset):
    def __init__(self, imgs, labels, transform):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]

        # Convert numpy to PIL
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Convert grayscale to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]

def get_dataloader(imgs, labels, batch_size=128):
    transform = get_transform()
    dataset = WrappedDataset(imgs, labels, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
