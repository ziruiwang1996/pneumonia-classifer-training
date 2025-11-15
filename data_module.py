
from torchvision import datasets, transforms
import lightning.pytorch as pl
import os
from torch.utils.data import DataLoader
import torch

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.482, 0.482, 0.482], [0.222, 0.222, 0.222])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.482, 0.482, 0.482], [0.222, 0.222, 0.222])
])

class ChestXRayDataModule(pl.LightningDataModule):
    """
    A LightningDataModule encapsulates all the steps involved in preparing data
    for a PyTorch model.
    """

    def __init__(self, data_dir, batch_size=64, device=None, num_workers=None):
        """
        Initializes the DataModule and stores key parameters.

        Args:
            data_dir (str): Directory where the data is stored.
            batch_size (int): Number of samples per batch in the DataLoader.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # Device can be 'cuda', 'mps', or 'cpu'. Used to select DataLoader options.
        self.device = device
        # If user doesn't provide num_workers, pick a reasonable default.
        self.num_workers = num_workers if num_workers is not None else max(0, (os.cpu_count() or 1) // 2)
        self.train_transform = TRAIN_TRANSFORM
        self.val_transform = VAL_TRANSFORM
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """
        Assigns the train and validation datasets.

        Args:
        stage (str, optional): The stage of training (e.g., 'fit', 'test').
                               The Lightning Trainer requires this argument, but it is not
                               utilized in this implementation as the setup logic is the
                               same for all stages. Defaults to None.
        """
        train_path = os.path.join(self.data_dir, "train")
        val_path = os.path.join(self.data_dir, "val")

        if not os.path.isdir(train_path):
            raise FileNotFoundError(f"Training directory not found: {train_path}")
        if not os.path.isdir(val_path):
            raise FileNotFoundError(f"Validation directory not found: {val_path}")

        self.train_dataset = datasets.ImageFolder(train_path, self.train_transform)
        self.val_dataset = datasets.ImageFolder(val_path, self.val_transform)

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        pin_memory = True if self.device == 'cuda' else False
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )        

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        pin_memory = True if self.device == 'cuda' else False
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        ) 
    
    def class_inx(self):
        """Returns the class to index mapping."""
        return self.train_dataset.class_to_idx

script_dir = os.path.dirname(__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
data_dir = os.path.abspath(os.path.join(script_dir, "chest_xray"))
dm = ChestXRayDataModule(data_dir, device=device, num_workers=0)
dm.setup()
print(dm.train_dataset.classes)
print(dm.class_inx())