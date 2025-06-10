import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from Network import Marepo_Regressor, DinoV2, MegaLoc, MLP
from Get_dataset import rgb_transforms
import torchvision.transforms.functional as TF
from My_Loss import my_loss
from Eval import evaluation_loop

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
        input_dim = 512
    else:
        second_encoder = None
        input_dim = 512

    mlp = MLP(input_dim=input_dim).to(device)

    optimizer = Adam(list(marepo.parameters()) + list(mlp.parameters()) + (list(second_encoder.parameters()) if second_encoder else []))
    loss_fn = my_loss()

    for epoch in range(epochs):
        marepo.train()
        if second_encoder:
            second_encoder.train()
        mlp.train()
        running_loss = 0.0
        for batch in train_dataloader:
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
            

            feats_pooled = torch.max(feats, dim=1)[0]  # oppure torch.mean(feat, dim=1)
            
            # MLP regression
            preds = mlp(feats_pooled)  # (B, 3)
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