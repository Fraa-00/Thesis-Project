import torch

def DinoV2():
    model_type = "dinov2_vitb14"
    model = torch.hub.load('facebookresearch/dinov2', model_type)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model

def MegaLoc():
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model

def Marepo():
    MAREPO_MODEL_PATH = "ace_encoder_pretrained.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(MAREPO_MODEL_PATH, map_location=device, weights_only=False)

def Ace():
       ACE_HEAD_PATH = "marepo_pretrained/marepo/marepo.pt"
       device = "cuda" if torch.cuda.is_available() else "cpu"
       model = torch.load(ACE_HEAD_PATH, map_location=device, weights_only=False)
       return model


