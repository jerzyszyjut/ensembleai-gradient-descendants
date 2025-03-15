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

from torch.utils.data import Dataset
from typing import Tuple, List
import time


TOKEN = "JL9uGkRYeY3vlJ0UV4XnpIghehTr3"              # Your token here
SUBMIT_URL = "149.156.182.9:6060/task-2/submit"
RESET_URL = "149.156.182.9:6060/task-2/reset"
QUERY_URL = "149.156.182.9:6060/task-2/query"
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
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)



def quering_example(images, outpath) -> Tuple[List, List]:
    image_data = []
    for i, img in enumerate(images):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        image_data.append(("files", (f"image{i}.png", img_byte_arr.getvalue(), "image/png")))

    response = requests.post(QUERY_URL, headers={"token": TOKEN}, files=image_data)

    if response.status_code == 200:
        buffer = io.BytesIO(response.content)
        np_array = np.load(buffer)
        print(np_array.shape)
        print(np_array)
        np.save("test.npy", np_array)
    # Store the output in a file.
    # Be careful to store all the outputs from the API since the number of queries is limited.
    with open(outpath, 'wb') as handle:
        pickle.dump(representation, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Restore the output from the file.
    with open(outpath, 'rb') as handle:
        representation = pickle.load(handle)

    print(len(representation))
    return representation


def submitting_example():

    # Create a dummy model
    model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 1024))

    path = 'dummy_submission.onnx'

    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
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


if __name__ == '__main__':
    dataset = torch.load("dataset.pt", weights_only=False)                   # Path to ModelStealingPub.pt
    # images = [dataset.imgs[idx] for idx in np.random.permutation(1000)]
    # print(images[0])
    # raise Exception

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1024)
    model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    indexes = np.random.permutation(3000)
    for begin in range(0, len(indexes), 1000):
        images = [dataset.imgs[idx] for idx in indexes[begin:begin+1000]]
        labels = quering_example(images, f"out_for_{begin}.pickle")
        running_loss = 0.0
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
        # image_for_victim = transforms_for_victim(image)
        image_for_stolen = transforms_for_stolen(images)
        # image_for_victim = image_for_victim.to(DEVICE, dtype=torch.float32)
        image_for_stolen = image_for_stolen.to(DEVICE, dtype=torch.float32)
        # step 5
        output_for_stolen = model(image_for_stolen)
        # output_for_victim = query_victim(image_for_victim, str(i))
        output_for_victim = torch.tensor(labels)
        # step 6
        loss = loss_fn(output_for_stolen, output_for_victim)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Batch {begin}, loss: {loss.item()}")
        time.sleep(60)
        # submitting_example()

