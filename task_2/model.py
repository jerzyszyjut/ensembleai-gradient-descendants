import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
import requests
from dataset import ModelStealingDataset
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor
import base64
import io
import pickle
import json
from transform_configs import get_random_resized_crop_config, get_jitter_color_config




TOKEN = ...                         # Your token here
QUERY_URL = "149.156.182.9:6060/task-2/query"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def query_victim(images):

    image_data = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        image_data.append(img_base64)

    payload = json.dumps(image_data)
    response = requests.get(QUERY_URL, headers={"token": TOKEN}, files={"file": payload})
    if response.status_code == 200:
        representation = response.json()["representations"]
    else:
        raise Exception(
            f"Model stealing failed. Code: {response.status_code}, content: {response.json()}"
        )
    # Store the output in a file.
    # Be careful to store all the outputs from the API since the number of queries is limited.
    with open('out.pickle', 'wb') as handle:
        pickle.dump(representation, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Restore the output from the file.
    with open('out.pickle', 'rb') as handle:
        representation = pickle.load(handle)

    return representation

# step 1
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50')
# victim_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
# load victim model from file
victim_model = torch.load('models/mnist_model.pt', weights_only=False)

model.to(DEVICE)
victim_model.to(DEVICE)

dataset = ModelStealingDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# step 2
running_loss = 0.0
for i, (image, _) in enumerate(dataloader):
    optimizer.zero_grad()
    # step 3
    transforms_for_victim = transforms.Compose([
        transforms.RandomResizedCrop(**get_random_resized_crop_config()),
        transforms.ColorJitter(**get_jitter_color_config()),
    ])
    # step 4
    image_for_victim = transforms_for_victim(image)
    image_for_stolen = image
    image_for_victim = image_for_victim.to(DEVICE, dtype=torch.float32)
    image_for_stolen = image_for_stolen.to(DEVICE, dtype=torch.float32)
    # step 5
    with torch.no_grad():
        output_for_stolen = model(image_for_stolen)
       # output_for_victim = query_victim(image_for_victim)
        output_for_victim = victim_model(image_for_victim)
    # step 6
    loss = loss_fn(output_for_stolen, output_for_victim)
    loss.requires_grad = True
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print(f"Batch {i+1}, loss: {loss.item()}")