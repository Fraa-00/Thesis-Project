import torch
import torchvision.transforms.functional as TF

def visualize_predictions(model, mlp, second_encoder, val_loader, device, num_samples=5):
    """
    Visualizza alcune predizioni con ground truth e immagini corrispondenti.
    
    Args:
        model: Modello principale (Marepo)
        mlp: MLP per la regressione finale
        second_encoder: Encoder secondario opzionale (DinoV2 o MegaLoc)
        val_loader: DataLoader per il validation set
        device: Device su cui eseguire le predizioni
        num_samples: Numero di campioni da visualizzare
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    if second_encoder:
        second_encoder.eval()
    mlp.eval()

    # Ottiene un batch di dati
    imgs, targets = next(iter(val_loader))
    imgs, targets = imgs.to(device), targets.to(device)
    
    with torch.no_grad():
        # Calcola le predizioni
        feat1 = model.get_features(TF.rgb_to_grayscale(imgs))
        feat1_flat = feat1.flatten(2).permute(0, 2, 1)
        if second_encoder:
            feat2 = second_encoder(imgs)
            feat2_flat = feat2.flatten(2).permute(0, 2, 1)
            feats = torch.cat([feat1_flat, feat2_flat], dim=-1)
        else:
            feats = feat1_flat

        pooled = torch.max(feats, dim=1)[0]
        preds = mlp(pooled)

    # Visualizza i risultati
    num_samples = min(num_samples, len(imgs))
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]

    # Denormalizza le immagini (sposta i tensori su GPU)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
    imgs_denorm = imgs * std + mean
    
    # Sposta su CPU solo quando necessario per visualizzazione
    for i in range(num_samples):
        # Visualizza l'immagine
        img = imgs_denorm[i].cpu().permute(1,2,0)
        axes[i].imshow(img.clamp(0,1))
        
        # Stampa predizioni e ground truth
        pred = preds[i].cpu().numpy()
        target = targets[i].cpu().numpy()
        
        title = f'Pred: (lat={pred[0]:.4f}, lon={pred[1]:.4f}, bearing={pred[2]:.1f}°)\n'
        title += f'GT: (lat={target[0]:.4f}, lon={target[1]:.4f}, bearing={target[2]:.1f}°)'
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
