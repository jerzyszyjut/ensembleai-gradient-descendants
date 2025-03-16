import base64
import io
import json
import numpy as np
import onnxruntime as ort
import pickle
import requests
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from transform_configs import get_random_resized_crop_config, get_jitter_color_config
import os
from torch.utils.data import Dataset
from typing import Tuple, List
import time
from scipy.optimize import curve_fit


TOKEN = "dJL9uGkRYeY3vlJ0UV4XnpIghehTr3"              # Your token here
SUBMIT_URL = "http://149.156.182.9:6060/task-2/submit"
RESET_URL = "http://149.156.182.9:6060/task-2/reset"
QUERY_URL = "http://149.156.182.9:6060/task-2/query"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


def get_dataloaders():
    # Load the full dataset from .pt file (an instance of TaskDataset)
    print("Loading custom dataset from:", "ModelStealingPub.pt")
    dataset_full = torch.load("dataset.pt", weights_only=False)

    vect = np.loadtxt("binary_vector.csv", delimiter=',', dtype=bool)
    dataset = [entry[1] for entry, k in zip(dataset_full, vect)]
    # quering_example(dataset[:1000])
    image_data = []
    for img in dataset:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        image_data.append(img_byte_arr.getvalue())
    quering_random(image_data[:1000])

# get_dataloaders()

def quering_random(images, output_path):
    image_data = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        image_data.append(img_byte_arr.getvalue())

    if os.path.isfile(output_path + ".npy"):
        print("Used cached data instead of downloading")
        return np.load(output_path + ".npy")
        
    files = [("files", ("image2.png", img, "image/png")) for img in image_data]
    response = requests.post(
        QUERY_URL,
        headers={"token": TOKEN},
        files=files
    )
    
    if response.status_code == 200:
        print("Response code: ", response.status_code)
        # print(response.headers)
        # print("Response content: ", response.content)
        buffer = io.BytesIO(response.content)
        np_array = np.load(buffer)
        np.save(output_path, np_array)
        np.save(output_path + "_images",  np.array(image_data))
        print(np_array.shape)
        print("==========================================================")
        print(np_array)
        print("==========================================================")
        time.sleep(60)
        return np_array
    else:
        print(response.content)


def submitting_example(model):

    # Create a dummy model
    # model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 1024))

    path = 'submission_2.onnx'

    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32).to(DEVICE),
        path,
        export_params=True,
        input_names=["x"],
    )

    # (these are being ran on the eval endpoint for every submission)
    with open(path, "rb") as f:
        model = f.read()
        try:
            stolen_model = ort.InferenceSession(model)
        except Exception as e:
            raise Exception(f"Invalid model, {e=}")
        try:
            out = stolen_model.run(
                None, {"x": np.random.randn(1, 3, 32, 32).astype(np.float32)}
            )[0][0]
        except Exception as e:
            raise Exception(f"Some issue with the input, {e=}")
        assert out.shape == (1024,), "Invalid output shape"

    response = requests.post(SUBMIT_URL, headers={"token": TOKEN}, files={"onnx_model": open(path, "rb")})
    print(response.status_code, response.text)

def exp_cost_function(epsilon, delta, alpha, beta):
    return delta * (np.exp(np.log(alpha/delta) * epsilon * beta) - 1)

if __name__ == '__main__':
    dataset = torch.load("dataset.pt", weights_only=False)

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1024)
    model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    batch_size = 100

    loaded_coefficients = np.load("linear_model.npy")
    linear_model = np.poly1d(loaded_coefficients) 

    indexes = np.random.permutation(range(5, 13000))
    for begin in range(0, len(indexes), 1000):
        special_images = [dataset.imgs[idx] for idx in range(5)]
        images = [dataset.imgs[idx] for idx in indexes[begin:begin+1000]]

        special_images = [dataset.imgs[idx] for idx in range(5)]
        for x in range(10):
            images[x*100:x*100+5] = special_images 
        images = [img.convert('RGB') for img in images]
        
        labels = quering_random(images, f"2_out_for_{begin}")
        for x in range(10):
            np.savetxt(f'labels/{begin+x*100+10}.csv', labels[x*100:x*100+5], delimiter=',')
        
        begin_iter = (begin % 1000) * 10

        running_loss = 0.0
        for idx in range(0, len(images), batch_size):
            optimizer.zero_grad()
            transforms_for_victim = transforms.Compose([
                transforms.RandomResizedCrop(**get_random_resized_crop_config()),
                transforms.ColorJitter(**get_jitter_color_config()),
            # transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]),
            ])
            transforms_for_stolen = transforms.Compose([
            # transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]),
            ])
            # step 4
            batch = images[idx:min(idx+batch_size, len(images))]
            batch = torch.stack([transforms.PILToTensor()(img) for img in batch])
            image_for_stolen = transforms_for_stolen(batch)
            image_for_stolen = image_for_stolen.to(DEVICE, dtype=torch.float32)
            # step 5
            output_for_stolen = model(image_for_stolen)
            output_for_victim = torch.tensor(labels[idx:min(idx+batch_size, len(images))]).to(DEVICE, dtype=torch.float32)
            # substract std from function
            sub = linear_model(begin_iter + idx)
            output_for_victim - sub
            # step 6
            loss = loss_fn(output_for_stolen, output_for_victim)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Batch {begin}, loss: {loss.item()}")
        # submitting_example()
    submitting_example(model)
