# Main

import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from Get_dataset import get_dataset, rgb_transforms
from Training import train_loop

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Set your dataset root directory here
    root_dir_train = "/kaggle/input/sf-xs/sf_xs/train"
    root_dir_val = "/kaggle/input/sf-xs/sf_xs/val"
    root_dir_test = "/kaggle/input/sf-xs/sf_xs/test"

    # Create dataset and dataloader
    train_dataset = get_dataset(root_dir_train, rgb_transform=rgb_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_dataset = get_dataset(root_dir_val, rgb_transform=rgb_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    # Run training loop for 1 epoch
    train_loop(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        use_second_encoder='dino',
        epochs=3,
        device="cuda"  # or "cpu"
    )



