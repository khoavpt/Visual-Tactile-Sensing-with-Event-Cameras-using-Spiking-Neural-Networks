import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class CustomSequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = self._make_dataset(self.root_dir)
        
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
        return sequence_data['sequence'], sequence_data['label']
    
def create_dataloader(
        data_dir: str,
        batch_size: int,
        num_workers: int = 1
):
    # Create the dataset
    dataset = CustomSequenceDataset(root_dir=data_dir)
   
    # Define the split ratio
    train_ratio = 0.8
    test_ratio = 0.2

    # Calculate the lengths for training and testing
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    test_size = dataset_size - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for the training and test sets
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  )
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 )

    # Check sizes of the splits
    print(f"Total number of sequences: {dataset_size}")
    print(f"Number of training sequences: {len(train_dataset)}")
    print(f"Number of test sequences: {len(test_dataset)}")

    # Iterate over the training DataLoader
    print("Training DataLoader:")
    for images, labels in train_dataloader:
        print(images.shape, labels.shape)
        break

    # Iterate over the test DataLoader
    print("Test DataLoader:")
    for images, labels in test_dataloader:
        print(images.shape, labels.shape)
        break

    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of test batches: {len(test_dataloader)}")
    return train_dataloader, test_dataloader