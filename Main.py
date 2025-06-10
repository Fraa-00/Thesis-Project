# Main

from torch.utils.data import DataLoader
from Get_dataset import get_dataset, rgb_transforms
from Training import train_loop

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
        epochs=10,
        device="cuda"  # or "cpu"
    )
    


