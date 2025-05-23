import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
import numpy as np


class FlipTransform():
    """Randomly flip a WHOLE sequence horizontally or vertically"""
    def __init__(self, p_horizontal=0.3, p_vertical=0.3):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical
    
    def __call__(self, sequence):
        """sequence: (sequence_length, channels, height, width)"""
        if torch.rand(1).item() < self.p_horizontal:
            sequence = torch.flip(sequence, dims=[-1])
        if torch.rand(1).item() < self.p_vertical:
            sequence = torch.flip(sequence, dims=[-2])
        return sequence

class CustomSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = self._make_dataset(self.root_dir)
        self.transform = transform
        
    def _make_dataset(self, dir):
        paths = []
        for file in os.listdir(dir):
            if file.endswith('.pt'):
                paths.append(os.path.join(dir, file))
        return paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            sequence (torch.Tensor): A sequence of frames of shape (sequence_length, channels, height, width)
            target (int): Target class index
        """
        path = self.samples[idx]
        sequence_data = torch.load(path)
        sequence, label = sequence_data['sequence'], sequence_data['label']
        
        if self.transform:
            sequence = torch.stack([self.transform(frame) for frame in sequence])
        
        return sequence, label

def create_dataloader(
        data_dir: str,
        batch_size: int,
        num_workers: int = 1
):
    # train_transform = FlipTransform(p_horizontal=0.2, p_vertical=0.2)
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')

    train_dataset = CustomSequenceDataset(root_dir=train_dir)
    test_dataset = CustomSequenceDataset(root_dir=test_dir)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  prefetch_factor=2)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 persistent_workers=True,
                                 prefetch_factor=2)
    
    print(f"Total number of sequences: {len(train_dataset) + len(test_dataset)}")
    print(f"Number of training sequences: {len(train_dataset)}")
    print(f"Number of test sequences: {len(test_dataset)}")

    # Count press and no press events in train and test sets
    def count_press_nopress(dataloader):
        press_count = 0
        nopress_count = 0
        for _, labels in dataloader:
            press_count += labels.sum().item()
            nopress_count += (labels.size(0) * labels.size(1)) - labels.sum().item()
        return press_count, nopress_count
    
    press_train, nopress_train = count_press_nopress(train_dataloader)
    press_test, nopress_test = count_press_nopress(test_dataloader)

    print(f"Number of press events in training set: {press_train}")
    print(f"Number of no press events in training set: {nopress_train}")

    print(f"Number of press events in test set: {press_test}")
    print(f"Number of no press events in test set: {nopress_test}")


    print("Training DataLoader:")
    for images, labels in train_dataloader:
        print(images.shape, labels.shape)
        break

    print("Test DataLoader:")
    for images, labels in test_dataloader:
        print(images.shape, labels.shape)
        break

    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of test batches: {len(test_dataloader)}")
    return train_dataloader, test_dataloader

def main():
    data_dir = r"data\seq_data_acc"
    
    batch_size = 32
    num_workers = 4

    train_dataloader, test_dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # Example usage of the dataloaders
    for images, labels in train_dataloader:
        print(images.shape, labels.shape)
        break

    for images, labels in test_dataloader:  
        print(images.shape, labels.shape)
        break

if __name__ == "__main__":
    main()