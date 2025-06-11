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
    use_second_encoder=None,  # 'dino' or 'megaloc' or None
    epochs=10,
    device='cuda',
    patience=3
):
    # Main encoder
    marepo = Marepo_Regressor(mean, num_head_blocks, use_homogeneous).to(device)
    # Optional second encoder
    if use_second_encoder == 'dino':
        second_encoder = DinoV2()
        input_dim = 512
    elif use_second_encoder == 'megaloc':
        second_encoder = MegaLoc()
        input_dim = 8960  # 8960 = 512 (DinoV2) + 8448 (MegaLoc)
    else:
        second_encoder = None
        input_dim = 512

    mlp = MLP(input_dim=input_dim).to(device)

    optimizer = Adam(list(marepo.parameters()) + list(mlp.parameters()) + (list(second_encoder.parameters()) if second_encoder else []))
    loss_fn = my_loss()
    best_val_loss = float('inf')
    bad_epochs = 0

    for epoch in range(epochs):
        marepo.train()
        if second_encoder:
            second_encoder.train()
        mlp.train()
        running_loss = 0.0
        for batch in train_dataloader:
            imgs, targets = batch  # imgs: (B, 3, H, W), targets: (B, N, 3)
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Main encoder features
            feat1 = marepo.get_features(TF.rgb_to_grayscale(imgs))
            feat1_flat = feat1.flatten(2).permute(0, 2, 1)  # (B, N, C)
            feat1_flat = torch.max(feat1_flat, dim=1)[0]  # (B, C)

            # Second encoder features (if any)
            if second_encoder:
                feat2 = second_encoder(imgs)
                print(feat2.shape)
                feats = torch.cat([feat1_flat, feat2], dim=1)
            else:
                feats = feat1_flat
        
            print(feats.shape)  # Debugging line to check shapes
            # MLP regression
            preds = mlp(feats)  # (B, 3)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        avg_val_loss = evaluation_loop(marepo, mlp, second_encoder, val_dataloader, loss_fn, device)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            bad_epochs = 0
            torch.save({
                'marepo': marepo.state_dict(),
                'mlp': mlp.state_dict(),
                **({'second': second_encoder.state_dict()} if second_encoder else {})
            }, 'best_model.pth')
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping attivato. Training interrotto.")
                break

    print("Training completato.")
    visualize_predictions(marepo, mlp, second_encoder, val_dataloader, device, num_samples=5)