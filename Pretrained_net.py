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


