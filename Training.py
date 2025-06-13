import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import random
import numpy as np
from Network import Marepo_Regressor, DinoV2, MegaLoc, MLP
from Get_dataset import rgb_transforms
import torchvision.transforms.functional as TF
from My_Loss import my_loss
from Eval import evaluation_loop
from Utils import visualize_predictions
import os
from VPR_Regressor import VPR_Regressor

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_loop(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    mean = 0,
    num_head_blocks=1,
    use_homogeneous=True,
    use_second_encoder=None,
    use_first_encoder=True,
    use_pose=True,
    epochs=10,
    device='cuda',
    patience=3,
    config_path=None
):
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    device = torch.device(device)
    
    # Carica la configurazione se fornita
    config = {}
    if config_path is not None:
        import json
        with open(config_path, "r") as f:
            config = json.load(f)

    model = VPR_Regressor(
        mean=mean,
        num_head_blocks=num_head_blocks,
        use_homogeneous=use_homogeneous,
        use_second_encoder=use_second_encoder,
        use_first_encoder=use_first_encoder,
        device=device,
        use_pose=True,
        config=config
    ).to(device)

    optimizer = Adam(model.get_trainable_parameters())
    loss_fn = my_loss()
    best_val_loss = float('inf')
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_dataloader:
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        avg_val_loss = evaluation_loop(model, val_dataloader, loss_fn, device)  # <-- Cambiato qui
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')  # <-- Cambiato qui
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping attivato. Training interrotto.")
                break

    print("Training completato.")

    # Visualizza le predizioni su alcuni sample di validazione
    visualize_predictions(model, val_dataloader, device, num_samples=5)  # <-- Cambiato qui
