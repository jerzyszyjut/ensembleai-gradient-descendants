import torch
from torchvision import models
import torch.nn as nn

allowed_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}



def load_model(model_name, model_path, device):
    try:
        model: nn.Module = allowed_models[model_name](weights=None)
        model.fc = nn.Linear(model.fc.weight.shape[1], 44)
    except Exception as e:
        print(1)
        raise Exception(
            f"Invalid model class, {e}, only {allowed_models.keys()} are allowed"
        )

    try:
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict, strict=True)
        model.eval()
    except Exception as e:
        print(2)
        raise Exception(f"Invalid model, {e}")

    model.to(device)
    return model