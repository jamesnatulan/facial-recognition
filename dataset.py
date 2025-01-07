# Contains the dataset class for the siamese network
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np


class SiameseDataset(Dataset):
    def __init__(self, image_pairs_csv, transform=None):
        # Load the data from the csv file
        self.data = pd.read_csv(image_pairs_csv)
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image paths
        img1_path = self.data.iloc[idx, 0]
        img2_path = self.data.iloc[idx, 1]

        # Load the images
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Apply the transformations
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        # Get the label
        label = self.data.iloc[idx, 2]
        label = torch.from_numpy(np.array([label], dtype=np.float32))
        return img1, img2, label
