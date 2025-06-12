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
    use_second_encoder=None,
    use_first_encoder=True,  # New parameter
    epochs=10,
    device='cuda',
    patience=3
):
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    device = torch.device(device)
    
    # Main encoder (optional now)
    first_encoder = Marepo_Regressor(mean, num_head_blocks, use_homogeneous).to(device) if use_first_encoder else None
    
    # Second encoder setup
    if use_second_encoder == 'dino':
        second_encoder = DinoV2().to(device)
        input_dim = 768  # DinoV2 output dimension
    elif use_second_encoder == 'megaloc':
        second_encoder = MegaLoc().to(device)
        input_dim = 8448  # MegaLoc output dimension
    else:
        second_encoder = None
        input_dim = 512

    # Adjust input dimension based on which encoders are used
    if use_first_encoder and second_encoder:
        input_dim = 512 + input_dim  # Combined features
    elif use_first_encoder:
        input_dim = 512  # Only first encoder
    # else keep input_dim as is (only second encoder)

    mlp = MLP(input_dim=input_dim, device=device)
    
    # Update optimizer parameters based on which encoders are used
    optimizer_params = []
    if use_first_encoder:
        optimizer_params.extend(first_encoder.parameters())
    if second_encoder:
        optimizer_params.extend(second_encoder.parameters())
    optimizer_params.extend(mlp.parameters())
    
    optimizer = Adam(optimizer_params)
    loss_fn = my_loss()
    best_val_loss = float('inf')
    bad_epochs = 0

    for epoch in range(epochs):
        if first_encoder:
            first_encoder.train()
        if second_encoder:
            second_encoder.train()
        mlp.train()
        running_loss = 0.0

        for batch in train_dataloader:
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Get features based on which encoders are used
            if use_first_encoder:
                feat1 = first_encoder.get_features(TF.rgb_to_grayscale(imgs))
                feat1_flat = feat1.flatten(2).permute(0, 2, 1)
                feat1_flat = torch.max(feat1_flat, dim=1)[0]
            
            if second_encoder:
                feat2 = second_encoder(imgs)
            
            # Combine features or use single encoder features
            if use_first_encoder and second_encoder:
                feats = torch.cat([feat1_flat, feat2], dim=-1)
            elif use_first_encoder:
                feats = feat1_flat
            else:
                feats = feat2
            
            preds = mlp(feats)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        avg_val_loss = evaluation_loop(first_encoder, mlp, second_encoder, val_dataloader, loss_fn, device)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            bad_epochs = 0
            torch.save({
                'first_encoder': first_encoder.state_dict(),
                'mlp': mlp.state_dict(),
                **({'second': second_encoder.state_dict()} if second_encoder else {})
            }, 'best_model.pth')
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping attivato. Training interrotto.")
                break

    print("Training completato.")
    visualize_predictions(first_encoder, mlp, second_encoder, val_dataloader, device, num_samples=5)