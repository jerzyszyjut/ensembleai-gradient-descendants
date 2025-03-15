from typing import Tuple
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
from torchmetrics import Accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class LinfPGDAttack:
    def __init__(self, model, epsilon, k, a, random_start):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start

    def perturb(self, x_nat, y):
        x = x_nat.clone().detach()
        if self.rand:
            x = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x = torch.clamp(x, 0, 1)
        
        x.requires_grad = True
        for _ in range(self.k):
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            self.model.zero_grad()
            loss.backward()
            x = x + self.a * x.grad.sign()
            x = torch.min(torch.max(x, x_nat - self.epsilon), x_nat + self.epsilon)
            x = torch.clamp(x, 0, 1)
            x = x.detach().clone().requires_grad_(True)
        return x.detach()

# Define the image transformation
t = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        # Optionally add normalization here if needed
        # transforms.Normalize((0.2980, 0.2962, 0.2987), (0.2886, 0.2875, 0.2889))
    ]
)

full_dataset = pickle.load(open("task_3/Train/data.pkl", "rb"))
full_dataset.transform = t

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
epochs = 100

attack = LinfPGDAttack(model=model, epsilon=0.05, k=5, a=0.01, random_start=True)
accuracies = []
losses = []

def train_one_epoch_adv():
    running_loss = 0.0
    model.train()
    start_time = time.time()
    counter = 0
    for data in train_dataloader:
        counter += 1
        if counter % 100 == 0:
            print(f"Batch: {counter} | Time taken: {time.time() - start_time:.2f}s")
        _id, inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_adv = attack.perturb(inputs, labels)

        optimizer.zero_grad()
        inputs_combined = torch.cat([inputs, inputs_adv], dim=0)
        labels_combined = torch.cat([labels, labels], dim=0)
        logits = model(inputs_combined)
        loss = loss_fn(logits, labels_combined)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    losses.append(running_loss)

    print(f"Time taken for training epoch: {time.time() - start_time:.2f}s")
    print(f"Running loss: {running_loss:.4f}")

def test_accuracy():
    model.eval()
    accuracy = Accuracy("multiclass", num_classes=10).to(device)
    with torch.no_grad():
        for data in test_dataloader:
            _id, inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            accuracy(outputs, labels)
    accuracies.append(accuracy.compute().cpu().item())
    print(f"Test Accuracy: {accuracies[-1]:.4f}")

# Optionally, load a pretrained model if available
# model.load_state_dict(torch.load("task_3/models/model_new_19.pth"))

for epoch in range(0, epochs):
    print(f"Epoch: {epoch}")
    train_one_epoch_adv()
    test_accuracy()
    torch.save(model.state_dict(), f"task_3/models/model_new_epoch_{epoch}.pth")

plt.plot(accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy vs Epoch")
plt.savefig("task_3/accuracy_plot.png")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.savefig("task_3/loss_plot.png")

torch.save(model.state_dict(), "task_3/Train/model_new_final.pth")
