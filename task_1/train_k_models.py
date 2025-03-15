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

BATCH_SIZE = 32
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100
K = 20 

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]),
])

def train_model(model, dataloader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for id, img, label, membership in dataloader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

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
        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        if accuracy == 100:
            # print("Finish training")
            break

    return model

if __name__ == "__main__":
    dataset: MembershipDataset = torch.load("dataset_0.pt", weights_only=False)
    dataset.transform = data_transforms

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    fold = 0

    for train_index, val_index in tqdm(kf.split(range(len(dataset)))):
        print(f"Training model {fold + 1}/{K}")
        
        # Create dataloaders for training and validation sets
        train_subset = torch.utils.data.Subset(dataset, train_index)
        # val_subset = torch.utils.data.Subset(dataset, val_index)
        
        train_dataloader = inference_dataloader(train_subset, BATCH_SIZE)
        # val_dataloader = inference_dataloader(val_subset, BATCH_SIZE)

        # Initialize model, loss function, optimizer
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 44)
        model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        trained_model = train_model(model, train_dataloader, criterion, optimizer, NUM_EPOCHS)

        # Save the trained model
        torch.save(trained_model.state_dict(), f"shadow_models/trained_model_fold_{fold}.pt")

        fold += 1

    print("Training complete for all folds.")