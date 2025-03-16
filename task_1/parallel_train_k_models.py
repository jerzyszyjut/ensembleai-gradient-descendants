from my_datasets import MembershipDataset, inference_dataloader
from load_my_model import load_model
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import threading

BATCH_SIZE = 32
DEVICE_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100
K = 20  # Number of splits for K-Fold Cross Validation

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]),
])

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=NUM_EPOCHS):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for id, img, label, membership in dataloader:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        epoch_loss = running_loss / len(dataloader)
        accuracy = correct / total * 100
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        if accuracy > 98:
            break

    return model

def train_on_device(device, folds, device_id):
    for fold, (train_index, val_index) in enumerate(folds):
        print(f"Training model {fold + 1}/{len(folds)} on GPU {device_id}")

        # Create dataloaders for training sets
        train_subset = torch.utils.data.Subset(dataset, train_index)
        train_dataloader = inference_dataloader(train_subset, BATCH_SIZE)

        # Initialize model, loss function, optimizer
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 44)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        trained_model = train_model(model, train_dataloader, criterion, optimizer, device, NUM_EPOCHS)

        # Save the trained model
        torch.save(trained_model.state_dict(), f"shadow_models/trained_model_gpu_{device_id}_fold_{fold}.pt")

    print(f"Training complete for GPU {device_id}.")

if __name__ == "__main__":
    dataset: MembershipDataset = torch.load("dataset_0.pt", weights_only=False)
    dataset.transform = data_transforms

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    fold = 0

    # Divide folds into two groups for two GPUs
    folds_device_0 = []
    folds_device_1 = []

    for train_index, val_index in kf.split(range(len(dataset))):
        if fold % 2 == 0:
            folds_device_0.append((train_index, val_index))
        else:
            folds_device_1.append((train_index, val_index))
        fold += 1

    # Create threads for training
    thread_0 = threading.Thread(target=train_on_device, args=(DEVICE_0, folds_device_0, 0))
    thread_1 = threading.Thread(target=train_on_device, args=(DEVICE_1, folds_device_1, 1))

    # Start threads
    thread_0.start()
    thread_1.start()

    # Wait for both threads to finish
    thread_0.join()
    thread_1.join()

    print("Training complete for all folds.")