import torch
import torchvision.transforms.functional as TF
from Network import Regressor, MLP
from Pretrained_net import DinoV2, MegaLoc
import json

class VPR_Regressor(torch.nn.Module):
    def __init__(
        self,
        mean,
        num_head_blocks=1,
        use_homogeneous=True,
        use_second_encoder=None,
        use_first_encoder=True,
        device='cuda'
    ):
        super().__init__()
        self.device = torch.device(device)
        self.use_first_encoder = use_first_encoder

        # Carica il file di configurazione JSON
        with open("nerf_focal_12T1R_256_homo.json", "r") as f:
            config = json.load(f)


        # Main encoder
        if use_first_encoder:
            self.first_encoder = Regressor.create_from_split_state_dict(
                encoder_state_dict= torch.load("marepo_pretrained/marepo/marepo.pt", weights_only=False),
                head_state_dict= torch.load("ace_encoder_pretrained.pt", weights_only=False),
                config=config
            ).to(self.device)
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
            feat1 = self.first_encoder.get_pose(TF.rgb_to_grayscale(imgs))
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
        if self.use_first_encoder:
            params += list(self.first_encoder.parameters())
        if self.second_encoder:
            params += list(self.second_encoder.parameters())
        params += list(self.mlp.parameters())
        return params