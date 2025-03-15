from typing import Tuple
import torch
from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms
from torchmetrics import Accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torchvision.models as models

class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)
    
t = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.2980, 0.2962, 0.2987), (0.2886, 0.2875, 0.2889))
    ]
)

dataset = pickle.load(open("task_3/Train/data.pkl", "rb"))
dataset.transform = t
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.weight.shape[1], 10)
model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
epochs = 25

def denorm(batch, mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()

    perturbations = epsilon * images.grad.sign()
    adversarial_images = images + perturbations
    adversarial_images = torch.clamp(adversarial_images, 0, 1)

    return adversarial_images

def train_one_epoch():
    running_loss = 0.0
    model.train()
    for data in dataloader:
        optimizer.zero_grad()

        id, inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        adv_fgsm = fgsm_attack(model, inputs, labels, 0.1)

        inputs = torch.cat((inputs, adv_fgsm), 0) 
        labels = torch.cat((labels, labels), 0)

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Running loss: {running_loss}")

def test_accuracy():
    model.eval()
    accuracy = Accuracy("multiclass", num_classes=10).to(device)
    fgsm_accuracy = Accuracy("multiclass", num_classes=10).to(device)
    with torch.no_grad():
        for data in dataloader:
            _id, inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            accuracy(outputs, labels)

            adv_fgsm = fgsm_attack(model, inputs, labels, 0.1)
            fgsm_outputs = model(adv_fgsm)
            fgsm_accuracy(fgsm_outputs, labels)

    print(f"Accuracy: {accuracy.compute()}")

for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    train_one_epoch()
    test_accuracy()
    torch.save(model.state_dict(), f"task_3/models/model_{epoch}.pth")

torch.save(model.state_dict(), "task_3/Train/model.pth")
