import pickle
from PIL import Image
import numpy as np
import os
from torchvision.transforms import PILToTensor, Compose
import torch


class ModelStealingDataset:

    def __init__(self, transforms=None):
        self.transform = transforms

        self.ids = []
        self.imgs = []
        self.labels = []

        for i in range(len(os.listdir("dataset/images"))):
            img = Image.open(f"dataset/images/{i}.png")
            with open(f"dataset/labels/{i}.txt", "r") as f:
                label = int(f.read())

            self.ids.append(i)
            self.imgs.append(img)
            self.labels.append(label)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        img = self.imgs[idx]
        label = self.labels[idx]

        if img.mode != "RGB":
            img = img.convert("RGB")

        im_arr = np.array(img)
        im_arr32 = im_arr.astype(np.float32)
        img = torch.tensor(im_arr32).permute(2, 0, 1)
        
        return img, label

# dataset = pickle.load(open("ModelStealingPub/data.pkl", "rb"))

# if not os.path.exists("dataset"):
#     os.makedirs("dataset")

# if not os.path.exists("dataset/labels"):
#     os.makedirs("dataset/labels")

# if not os.path.exists("dataset/images"):
#     os.makedirs("dataset/images")

# for i, (img, label) in enumerate(dataset):
#     img.save(f"dataset/images/{i}.png")
#     with open(f"dataset/labels/{i}.txt", "w") as f:
#         f.write(str(label))