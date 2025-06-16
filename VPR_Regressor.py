import torch
import torchvision.transforms.functional as TF
from Network import Marepo_Regressor, MLP
from Pretrained_net import DinoV2, MegaLoc, Marepo,  Ace

class VPR_Regressor(torch.nn.Module):
    def __init__(
        self,
        mean=torch.zeros(1, 3, 1, 1),
        num_head_blocks=1,
        use_homogeneous=True,
        use_second_encoder=None,
        use_first_encoder=True,
        device='cuda'
    ):
        super().__init__()
        self.device = torch.device(device)
        self.use_first_encoder = use_first_encoder

        # Main encoder
        if use_first_encoder:
            self.first_encoder_back = Ace()
            self.first_encoder_head = Marepo()
        else:
            self.first_encoder = None

        # Second encoder
        if use_second_encoder == 'dino':
            self.second_encoder = DinoV2().to(self.device)
            second_dim = 768
        elif use_second_encoder == 'megaloc':
            self.second_encoder = MegaLoc().to(self.device)
            second_dim = 8448
        else:
            self.second_encoder = None
            second_dim = 6

        # Input dimension for MLP
        if use_first_encoder and self.second_encoder:
            input_dim = 6 + second_dim
        elif use_first_encoder:
            input_dim = 6
        else:
            input_dim = second_dim

        self.mlp = MLP(input_dim=input_dim, device=self.device).to(self.device)

    def forward(self, imgs):
        feats = []
        if self.use_first_encoder:
            feat1 = self.first_encoder_back(TF.rgb_to_grayscale(imgs))
            feat1 = self.first_encoder_head(features)
            # Ottieni feature map (batch, 512, H, W) -> pooling -> (batch, 512)
            feats.append(feat1)
        if self.second_encoder:
            feat2 = self.second_encoder(imgs)
            feats.append(feat2)
        if len(feats) > 1:
            feats = torch.cat(feats, dim=-1)
        else:
            feats = feats[0]
        preds = self.mlp(feats)
        return preds

    def get_trainable_parameters(self):
        params = []
        if self.first_encoder:
            params += list(self.first_encoder.parameters())
        if self.second_encoder:
            params += list(self.second_encoder.parameters())
        params += list(self.mlp.parameters())
        return params