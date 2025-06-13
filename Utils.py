import torch
import matplotlib.pyplot as plt

def visualize_predictions(first_encoder, mlp, second_encoder, dataloader, device, num_samples=5):
    first_encoder.eval() if first_encoder else None
    second_encoder.eval() if second_encoder else None
    mlp.eval()
    shown = 0

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Feature extraction
            if first_encoder:
                import torchvision.transforms.functional as TF
                feat1 = first_encoder.get_features(TF.rgb_to_grayscale(imgs))
                feat1_flat = feat1.flatten(2).permute(0, 2, 1)
                feat1_flat = torch.max(feat1_flat, dim=1)[0]
            if second_encoder:
                feat2 = second_encoder(imgs)
            if first_encoder and second_encoder:
                feats = torch.cat([feat1_flat, feat2], dim=-1)
            elif first_encoder:
                feats = feat1_flat
            else:
                feats = feat2

            preds = mlp(feats).cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i in range(preds.shape[0]):
                if shown >= num_samples:
                    return
                print(f"Sample {shown+1}:")
                print(f"  Prediction: lat={preds[i][0]:.5f}, lon={preds[i][1]:.5f}, θ={preds[i][2]:.2f}")
                print(f"  Ground Truth: lat={targets_np[i][0]:.5f}, lon={targets_np[i][1]:.5f}, θ={targets_np[i][2]:.2f}")
                print("-" * 40)
                shown += 1