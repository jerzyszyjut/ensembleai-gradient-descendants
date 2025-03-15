from my_datasets import MembershipDataset, inference_dataloader
from load_my_model import load_model
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

BATCH_SIZE = 32
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]),
])

if __name__ == "__main__":
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 44)
    model.to(DEVICE)
    dataset: MembershipDataset = torch.load("dataset_0.pt", weights_only=False)
    dataset.transform = data_transforms
    dataloader = inference_dataloader(dataset, BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for id, img, label, membership in tqdm(dataloader):
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()  # Wyzeruj gradienty

            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()  # Wsteczna propagacja błędu
            optimizer.step()  # Aktualizacja wag

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        epoch_loss = running_loss / len(dataloader)
        accuracy = correct / total * 100
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        if accuracy == 100:
            print("Finish training")
            break

    torch.save(model.state_dict(), "trained_model.pt")