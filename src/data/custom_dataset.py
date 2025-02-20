import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
import numpy as np

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
    # train_transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    # ])
    
    dataset = CustomSequenceDataset(root_dir=data_dir)
   
    # Extract labels for stratification
    labels = [torch.load(path)['label'] for path in dataset.samples]
    labels = np.array([torch.sum(label).item() for label in labels])  # Sum of press events in each label

    # Stratified split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(np.zeros(len(labels)), labels))

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    # train_dataset.dataset.transform = train_transform

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    print(f"Total number of sequences: {len(dataset)}")
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

    train_press, train_nopress = count_press_nopress(train_dataloader)
    test_press, test_nopress = count_press_nopress(test_dataloader)

    print(f"Training set - Press: {train_press}, No Press: {train_nopress}")
    print(f"Test set - Press: {test_press}, No Press: {test_nopress}")

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