import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from Network import Marepo_Regressor, DinoV2, MegaLoc, MLP
from My_DataLoader import rgb_transforms
import torchvision.transforms.functional as TF
from My_Loss import my_loss

def train_loop(
    dataloader: DataLoader,
    mean = 0,
    num_head_blocks=1,
    use_homogeneous=True,
    use_second_encoder=None,  # 'dino' or 'megaloc' or None
    epochs=10,
    device='cuda'
):
    # Main encoder
    marepo = Marepo_Regressor(mean, num_head_blocks, use_homogeneous).to(device)
    # Optional second encoder
    if use_second_encoder == 'dino':
        second_encoder = DinoV2()
    elif use_second_encoder == 'megaloc':
        second_encoder = MegaLoc()
    else:
        second_encoder = None

    # MLP input dim: sum of feature dims if using two encoders, else just Marepo
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 64, 64).to(device)
        feat1 = marepo.get_features(dummy)
        feat1_flat = feat1.flatten(2).permute(0, 2, 1)  # (B, N, C)
        dim1 = feat1_flat.shape[-1]
        if second_encoder:
            feat2 = second_encoder(dummy.repeat(1,3,1,1))  # Assume 3ch input for Dino/MegaLoc
            feat2_flat = feat2.flatten(2).permute(0, 2, 1)
            dim2 = feat2_flat.shape[-1]
        else:
            dim2 = 0
    
    print(dim1, dim2)

    mlp = MLP(input_dim=dim1+dim2).to(device)

    optimizer = Adam(list(marepo.parameters()) + list(mlp.parameters()) + (list(second_encoder.parameters()) if second_encoder else []))
    loss_fn = my_loss()

    for epoch in range(epochs):
        marepo.train()
        if second_encoder:
            second_encoder.train()
        mlp.train()
        for batch in dataloader:
            imgs, targets = batch  # imgs: (B, 1, H, W), targets: (B, N, 3)
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Main encoder features
            feat1 = marepo.get_features(TF.rgb_to_grayscale(imgs))
            feat1_flat = feat1.flatten(2).permute(0, 2, 1)  # (B, N, C)

            # Second encoder features (if any)
            if second_encoder:
                feat2 = second_encoder(imgs)
                feat2_flat = feat2.flatten(2).permute(0, 2, 1)
                feats = torch.cat([feat1_flat, feat2_flat], dim=-1)
            else:
                feats = feat1_flat
            
            print(feats.shape, targets.shape)

            feats_pooled = torch.max(feats, dim=1)[0]  # oppure torch.mean(feat, dim=1)
            
            # MLP regression
            preds = mlp(feats_pooled)  # (B, 3)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")