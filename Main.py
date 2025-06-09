# Main

from torch.utils.data import DataLoader
from My_DataLoader import Images, grayscale_transforms
from Training import train_loop

if __name__ == "__main__":
    # Set your dataset root directory here
    root_dir = "path/to/your/images"

    # Create dataset and dataloader
    dataset = Images(root_dir, grayscale_transform=grayscale_transforms())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Run training loop for 1 epoch
    train_loop(
        dataloader=dataloader,
        epochs=1,
        device="cuda"  # or "cpu"
    )

