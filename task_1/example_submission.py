import pandas as pd
import requests
import torch
import torch.nn as nn

from torchvision.models import resnet18
from my_example import membership_inference_attack
from my_datasets import MembershipDataset, inference_dataloader, TaskDataset
from load_my_model import load_model
import os
from tqdm import tqdm

from torchvision import transforms

from torch.utils.data import Subset
import random

TOKEN = "dJL9uGkRYeY3vlJ0UV4XnpIghehTr3"                        # Your token here
URL = "149.156.182.9:6060/task-1/submit"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
MEMBERSHIP_DATASET_PATH = "priv_out.pt"       # Path to priv_out_.pt
MIA_CKPT_PATH = "01_MIA_69.pt"                 # Path to 01_MIA_69.pt

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]),
])

def load_reference_models(folder):
    reference_models = []
    for filename in os.listdir(folder):
        if filename.endswith(".pt"):
            model = resnet18()
            model.fc = nn.Linear(model.fc.in_features, 44)
            model.load_state_dict(torch.load(os.path.join(folder, filename)))
            model.to(DEVICE)
            model.eval()
            reference_models.append(model)
    return reference_models

def get_random_subset(dataset, num_samples=10):
    indices = random.sample(range(len(dataset)), num_samples)
    return Subset(dataset, indices)

def membership_prediction(model, a=0.7962450055858669, gamma=1.201362676038642, z_nr=12): # a=0.4316950209139891, gamma=1.0158915284568744 
    dataset: MembershipDataset = torch.load(MEMBERSHIP_DATASET_PATH, weights_only=False)
    dataset_z: MembershipDataset = torch.load("dataset_0.pt", weights_only=False)
    
    
    dataset.transform = data_transforms
    dataset_z.transform = data_transforms

    dataloader = inference_dataloader(dataset, BATCH_SIZE)
    k = 20
    reference_models = load_reference_models("shadow_models")

    outputs_list = []

    for id, img, label, _ in tqdm(dataloader):

        my_dataset_z = get_random_subset(dataset_z, z_nr)
        dataloader_z = inference_dataloader(my_dataset_z, BATCH_SIZE)

        with torch.no_grad():
            membership_output = membership_inference_attack(model, img, reference_models, k, a, gamma, dataloader_z, DEVICE)
            print(membership_output)

        outputs_list.append(membership_output)

    return pd.DataFrame(
        {
            "ids": dataset.ids,
            "score": outputs_list,
        }
    )


if __name__ == '__main__':
    model = load_model(model_name="resnet18", model_path=MIA_CKPT_PATH, device=DEVICE)                 # Insert model name and path to your model
    preds = membership_prediction(model)
    preds.to_csv("results/submission.csv", index=False)

    # result = requests.post(
    #     URL,
    #     headers={"token": TOKEN},
    #     files={
    #         "csv_file": ("submission.csv", open("./submission.csv", "rb"))
    #     }
    # )

    # print(result.status_code, result.text)
