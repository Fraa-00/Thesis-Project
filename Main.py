# Main

from torch.utils.data import DataLoader
from My_DataLoader import Images, rgb_transforms
from Training import train_loop

if __name__ == "__main__":
    # Set your dataset root directory here
    root_dir = "/kaggle/input/sf-xs/sf_xs/train"

    # Create dataset and dataloader
    dataset = Images(root_dir, rgb_transform=rgb_transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Run training loop for 1 epoch
    train_loop(
        dataloader=dataloader,
        epochs=1,
        device="cuda"  # or "cpu"
    )

