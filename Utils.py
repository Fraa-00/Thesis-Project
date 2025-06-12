import torch
import torchvision.transforms.functional as TF

def visualize_predictions(first_encoder, mlp, second_encoder, val_loader, device, num_samples=5, save_path='predictions.png'):
    """
    Visualizza alcune predizioni con ground truth e immagini corrispondenti e salva il risultato.
    
    Args:
        first_encoder: Modello principale (Marepo)
        mlp: MLP per la regressione finale
        second_encoder: Encoder secondario opzionale (DinoV2 o MegaLoc)
        val_loader: DataLoader per il validation set
        device: Device su cui eseguire le predizioni
        num_samples: Numero di campioni da visualizzare
        save_path: Percorso dove salvare l'immagine risultante
    """
    import matplotlib.pyplot as plt
    
    if first_encoder:
        first_encoder.eval()
    if second_encoder:
        second_encoder.eval()
    mlp.eval()

    # Ottiene un batch di dati
    imgs, targets = next(iter(val_loader))
    imgs, targets = imgs.to(device), targets.to(device)
    
    with torch.no_grad():
        if first_encoder:
            # Calcola le predizioni
            feat1 = first_encoder.get_features(TF.rgb_to_grayscale(imgs))
            feat1_flat = feat1.flatten(2).permute(0, 2, 1)
            feat1_flat = torch.max(feat1_flat, dim=1)[0]  # (B, C)
        
        if second_encoder:
            feat2 = second_encoder(imgs)  # Already (B, C)
        
        if first_encoder and second_encoder:
            feats = torch.cat([feat1_flat, feat2], dim=-1)
        elif first_encoder:
            feats = feat1_flat
        else:
            feats = feat2

        preds = mlp(feats)

    # Visualizza i risultati
    num_samples = min(num_samples, len(imgs))
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]

    # Denormalizza le immagini
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
    imgs_denorm = imgs * std + mean
    
    for i in range(num_samples):
        img = imgs_denorm[i].cpu().permute(1,2,0)
        axes[i].imshow(img.clamp(0,1))
        
        pred = preds[i].cpu().numpy()
        target = targets[i].cpu().numpy()
        
        title = f'Pred: (lat={pred[0]:.4f}, lon={pred[1]:.4f}, bearing={pred[2]:.1f}°)\n'
        title += f'GT: (lat={target[0]:.4f}, lon={target[1]:.4f}, bearing={target[2]:.1f}°)'
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    
    # Save the figure instead of showing it
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
    
    # Print message about where the plot was saved
    print(f"Predictions visualization saved to {save_path}")
