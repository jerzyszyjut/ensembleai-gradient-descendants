import pandas as pd
import requests
import torch
import torch.nn as nn
import optuna
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18
from my_example import membership_inference_attack
from my_datasets import MembershipDataset, inference_dataloader
from load_my_model import load_model
import os
from tqdm import tqdm

from torchvision import transforms

from torch.utils.data import Subset
import random

TOKEN = "dJL9uGkRYeY3vlJ0UV4XnpIghehTr3"                        # Your token here
URL = "149.156.182.9:6060/task-1/submit"

BATCH_SIZE = 1
MEMBERSHIP_DATASET_PATH = "pub.pt"       # Path to priv_out_.pt
MIA_CKPT_PATH = "01_MIA_69.pt"                 # Path to 01_MIA_69.pt

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]),
])

import threading

my_gpu_queue = [0] * torch.cuda.device_count()

lock = threading.Lock()

def get_gpu():
    with lock:
        for i in range(len(my_gpu_queue)):
            if my_gpu_queue[i] == 0: 
                my_gpu_queue[i] = 1
                return i
    return None

def free_gpu(gpu_id):
    with lock:
        my_gpu_queue[gpu_id] = 0

def load_reference_models(folder, DEVICE):
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

def membership_prediction(model, a, gamma, z_nr, DEVICE):
    

    dataset: MembershipDataset = torch.load("pub.pt", weights_only=False)
    dataset_z: MembershipDataset = torch.load("dataset_0.pt", weights_only=False)
    
    
    dataset = get_random_subset(dataset, 500)

    dataset.transform = data_transforms
    dataset_z.transform = data_transforms

    dataloader = inference_dataloader(dataset, BATCH_SIZE)
    k = 20
    reference_models = load_reference_models("shadow_models", DEVICE)

    outputs_list = []

    y_true = []

    for id, img, label, membership in dataloader:

        my_dataset_z = get_random_subset(dataset_z, z_nr)
        dataloader_z = inference_dataloader(my_dataset_z, BATCH_SIZE)

        with torch.no_grad():
            membership_output = membership_inference_attack(model, img, reference_models, k, a, gamma, dataloader_z, DEVICE)
            outputs_list.append(membership_output)
            y_true.append(membership.item())

    return y_true, outputs_list

def calculate_auc_roc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)

def objective(trial):
    gpu_id = get_gpu()
    if gpu_id is None:
        raise RuntimeError("No free GPU available")

    a = trial.suggest_float('a', 0.01, 1.0)
    gamma = trial.suggest_float('gamma', 1.0, 2.0)
    z_nr = trial.suggest_int('z_nr', 5, 20)

    DEVICE = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    model = load_model(model_name="resnet18", model_path=MIA_CKPT_PATH, device=DEVICE)
    
    y_true, y_scores = membership_prediction(model, a, gamma, z_nr, DEVICE)

    free_gpu(gpu_id)

    auc = calculate_auc_roc(y_true, y_scores)
    return auc

if __name__ == '__main__':
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=2)

    best_trial = study.best_trial
    best_a = best_trial.params['a']
    best_gamma = best_trial.params['gamma']
    best_z_nr = best_trial.params['z_nr']

    print(f"Best AUC: {best_trial.value} with a={best_a} and gamma={best_gamma} and z_nr={best_z_nr}")

