import pandas as pd
import requests
import torch
import torch.nn as nn

from torchvision.models import resnet18
from my_example import membership_inference_attack
from my_datasets import MembershipDataset, inference_dataloader
from load_my_model import load_model



TOKEN = "dJL9uGkRYeY3vlJ0UV4XnpIghehTr3"                        # Your token here
URL = "149.156.182.9:6060/task-1/submit"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
MEMBERSHIP_DATASET_PATH = "pub.pt"       # Path to priv_out_.pt
MIA_CKPT_PATH = "01_MIA_69.pt"                 # Path to 01_MIA_69.pt

def load_reference_models(k):
    reference_models = []
    for _ in range(k):
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 44)
        reference_models.append(model)
    return reference_models

def membership_prediction(model):
    dataset: MembershipDataset = torch.load(MEMBERSHIP_DATASET_PATH, weights_only=False)
    dataloader = inference_dataloader(dataset, BATCH_SIZE)

    outputs_list = []
    k = 10

    reference_models = load_reference_models(k)

    for id, img, label, membership in dataloader:
        print(id)
        print(img)
        print(label)
        print(membership)
        img = img.to(DEVICE)

        with torch.no_grad():
            membership_output = model(img)
            # print(membership_output)

        outputs_list += membership_output.tolist()
        break

    return pd.DataFrame(
        {
            "ids": dataset.ids,
            "score": outputs_list,
        }
    )


if __name__ == '__main__':
    model = load_model(model_name="resnet18", model_path=MIA_CKPT_PATH, DEVICE)                 # Insert model name and path to your model
    preds = membership_prediction(model)
    preds.to_csv("submission.csv", index=False)

    # result = requests.post(
    #     URL,
    #     headers={"token": TOKEN},
    #     files={
    #         "csv_file": ("submission.csv", open("./submission.csv", "rb"))
    #     }
    # )

    # print(result.status_code, result.text)
