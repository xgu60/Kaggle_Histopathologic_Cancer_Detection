import numpy as np
import pandas as pd
import os, os.path
from skimage import io, transform
from torch.utils.data import Dataset


class HistoDataset(Dataset):
    """kaggle histo dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with id and label.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.id = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.id.iloc[idx, 1]+'.tif')
        image = io.imread(img_name)
        label = self.id.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
