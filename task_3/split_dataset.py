import torch
import torchvision
import torchvision.transforms as transforms
from torchmetrics import Accuracy
import pickle
import os

class TaskDataset:
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        img = self.imgs[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return _id, img, label  
    
transform = transforms.Compose(
    [transforms.ToTensor()]
)
 
dataset = pickle.load(open("task_3/Train/data.pkl", "rb"))

train_part = int(len(dataset) * 0.98)

os.makedirs(f"task_3/dataset2", exist_ok=True)
os.makedirs(f"task_3/dataset2/train", exist_ok=True)
os.makedirs(f"task_3/dataset2/val", exist_ok=True)

for i in range(train_part):
    _id, img, label = dataset[i]
    os.makedirs(f"task_3/dataset2/train/{label}", exist_ok=True)
    img.save(f"task_3/dataset2/train/{label}/{_id}.png")

for i in range(train_part, len(dataset)):
    _id, img, label = dataset[i]
    os.makedirs(f"task_3/dataset2/val/{label}", exist_ok=True)
    img.save(f"task_3/dataset2/val/{label}/{_id}.png")
