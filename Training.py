import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from Network import Marepo_Regressor, DinoV2, MegaLoc, MLP
from My_DataLoader import rgb_transforms
import torchvision.transforms.functional as TF
from My_Loss import my_loss

def train_loop(
    train_loader: DataLoader,
    val_loader: DataLoader,
    mean=0,
    num_head_blocks=1,
    use_homogeneous=True,
    use_second_encoder=None,  # 'dino', 'megaloc', or None
    epochs=10,
    patience=3,
    device='cuda'
):
    """
    Esempio di training con validazione.

    Parametri:
      - train_loader: DataLoader per il training set
      - val_loader: DataLoader per il validation set
      - mean, num_head_blocks, use_homogeneous: parametri di Marepo_Regressor
      - use_second_encoder: None | 'dino' | 'megaloc'
      - epochs: numero massimo di epoche
      - patience: epoche di attesa per early stopping
      - device: 'cuda' o 'cpu'
    """
    # Inizializza encoder principale e opzionale
    marepo = Marepo_Regressor(mean, num_head_blocks, use_homogeneous).to(device)
    if use_second_encoder == 'dino':
        second_encoder = DinoV2().to(device)
    elif use_second_encoder == 'megaloc':
        second_encoder = MegaLoc().to(device)
    else:
        second_encoder = None

    # Calcola dimensione input per MLP
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 64, 64).to(device)
        feat1 = marepo.get_features(dummy)
        feat1_flat = feat1.flatten(2).permute(0, 2, 1)
        dim1 = feat1_flat.shape[-1]
        dim2 = 0
        if second_encoder:
            feat2 = second_encoder(dummy.repeat(1, 3, 1, 1))
            feat2_flat = feat2.flatten(2).permute(0, 2, 1)
            dim2 = feat2_flat.shape[-1]

    mlp = MLP(input_dim=dim1 + dim2).to(device)

    # Ottimizzatore e loss
    params = list(marepo.parameters()) + list(mlp.parameters())
    if second_encoder:
        params += list(second_encoder.parameters())
    optimizer = Adam(params)
    loss_fn = my_loss()

    best_val_loss = float('inf')
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        # ----- Training -----
        marepo.train()
        if second_encoder:
            second_encoder.train()
        mlp.train()

        running_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            # Estrai features
            feat1 = marepo.get_features(TF.rgb_to_grayscale(imgs))
            feat1_flat = feat1.flatten(2).permute(0, 2, 1)
            if second_encoder:
                feat2 = second_encoder(imgs)
                feat2_flat = feat2.flatten(2).permute(0, 2, 1)
                feats = torch.cat([feat1_flat, feat2_flat], dim=-1)
            else:
                feats = feat1_flat

            pooled = torch.max(feats, dim=1)[0]
            preds = mlp(pooled)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # ----- Validazione -----
        marepo.eval()
        if second_encoder:
            second_encoder.eval()
        mlp.eval()

        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                feat1 = marepo.get_features(TF.rgb_to_grayscale(imgs))
                feat1_flat = feat1.flatten(2).permute(0, 2, 1)
                if second_encoder:
                    feat2 = second_encoder(imgs)
                    feat2_flat = feat2.flatten(2).permute(0, 2, 1)
                    feats = torch.cat([feat1_flat, feat2_flat], dim=-1)
                else:
                    feats = feat1_flat

                pooled = torch.max(feats, dim=1)[0]
                preds = mlp(pooled)
                val_loss += loss_fn(preds, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Early stopping
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
