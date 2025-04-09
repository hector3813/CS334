"""
Dogs Dataset
    Class wrapper for interfacing with the dataset of dog images
"""
import os
import numpy as np
import pandas as pd
import torch
import skimage as ski
from torch.utils.data import Dataset, DataLoader
from utils import config
import utils

def resize(X):
    """
    Resizes the data partition X to the size specified in the config file.

    Returns:
        the resized images as a numpy array.
    """
    image_dim = config('image_dim')
    
    image_size = (image_dim, image_dim)
    resized = []
    for i in range(X.shape[0]):
        xi = ski.transform.resize(X[i], image_size, preserve_range=True)
        resized.append(xi)
    resized = np.array(resized)

    return resized

class ImageStandardizer(object):
    """
    Channel-wise standardization for batch of images to mean 0 and variance 1. 
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.
    
    X has shape (N, image_height, image_width, color_channel)
    """
    def __init__(self):
        super().__init__()
        self.image_mean = None
        self.image_std = None
    
    def fit(self, X):
        self.image_mean = np.mean(X, axis=(0,1,2))
        self.image_std = np.std(X, axis=(0,1,2))
    
    def transform(self, X):
        return (X - self.image_mean) / self.image_std

class DogsDataset(Dataset):

    def __init__(self, partition, num_classes=10):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()

        if partition not in ['train', 'val', 'test']:
            raise ValueError('Partition {} does not exist'.format(partition))
        
        np.random.seed(0)
        self.partition = partition
        self.num_classes = num_classes
        
        # Load in all the data we need from disk
        self.metadata = pd.read_csv(config('csv_file'), index_col=0)
        self.X, self.y = self._load_data()
        self.X = resize(self.X)

        self.semantic_labels = dict(zip(
            self.metadata['numeric_label'].dropna().astype(int),
            self.metadata['semantic_label'].dropna(),
        ))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def _load_data(self):
        """
        Loads a single data partition from file.
        """
        print("Loading %s..." % self.partition)
        
        if self.partition == 'test':
            if self.num_classes == 5:
                df = self.metadata[self.metadata.partition == self.partition]
            elif self.num_classes == 10:
                df = self.metadata[self.metadata.partition.isin([self.partition, ' '])]
            else:
                raise ValueError('Unsupported test partition: num_classes must be 5 or 10')
        else:
            df = self.metadata[
                (self.metadata.numeric_label < self.num_classes) &
                (self.metadata.partition == self.partition)
            ]
        
        X, y = [], []
        for i, row in df.iterrows():
            image = ski.io.imread(os.path.join(config('image_path'), row['filename']))
            label = row['numeric_label']
            X.append(image)
            y.append(label)
        
        return np.array(X), np.array(y)

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.semantic_labels[numeric_label]

def get_train_val_test_loaders(num_classes):
    tr, va, te, _ = get_train_val_dataset(num_classes=num_classes)
    
    batch_size = config('cnn.batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader, te_loader, tr.get_semantic_label

def get_train_val_dataset(num_classes=10):
    tr = DogsDataset('train', num_classes)
    va = DogsDataset('val', num_classes)
    te = DogsDataset('test', num_classes)
    
    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)
    
    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0,3,1,2)
    va.X = va.X.transpose(0,3,1,2)
    te.X = te.X.transpose(0,3,1,2)

    # Move data to the device
    device = utils.get_device()
    tr.X = torch.from_numpy(tr.X).float().to(device)
    va.X = torch.from_numpy(va.X).float().to(device)
    te.X = torch.from_numpy(te.X).float().to(device)
    tr.y = torch.from_numpy(tr.y).long().to(device)
    va.y = torch.from_numpy(va.y).long().to(device)
    te.y = torch.from_numpy(te.y).long().to(device)
    
    return tr, va, te, standardizer

if __name__ == '__main__':
    np.set_printoptions(precision=2)
    tr, va, te, standardizer = get_train_val_dataset()
    print("Train:\t", len(tr.X))
    print("Val:\t", len(va.X))
    print("Test:\t", len(te.X))
    print("ImageStandardizer image_mean:", standardizer.image_mean)
    print("ImageStandardizer image_std: ", standardizer.image_std)
