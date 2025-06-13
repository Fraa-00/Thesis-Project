import torch
import matplotlib.pyplot as plt

def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    shown = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs).cpu().numpy()
            targets_np = targets.cpu().numpy()
            for i in range(preds.shape[0]):
                if shown >= num_samples:
                    return
                print(f"Sample {shown+1}:")
                print(f"  Prediction: lat={preds[i][0]:.5f}, lon={preds[i][1]:.5f}, θ={preds[i][2]:.2f}")
                print(f"  Ground Truth: lat={targets_np[i][0]:.5f}, lon={targets_np[i][1]:.5f}, θ={targets_np[i][2]:.2f}")
                print("-" * 40)
                shown += 1