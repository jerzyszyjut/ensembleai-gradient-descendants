import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm 

def Pr(sample, model, device):
    sample = sample.to(device)
    with torch.no_grad():
        output = model(sample)
    return torch.softmax(output, dim=1).max().item()

def membership_inference_attack(model, x, shadow_models, k, a, gamma, Z, device): # gamma 2
    C = 0
    Pr_x_out = sum(Pr(x, shadow_model, device) for shadow_model in shadow_models) / k
    Pr_x = 0.5 * ((1 + a) * Pr_x_out + (1 - a))

    Ratio_x = Pr(x, model, device) / Pr_x

    for id, z, label, membership in Z:
        Pr_z = sum(Pr(z, shadow_model, device) for shadow_model in shadow_models) / k
        Ratio_z = Pr(z, model, device) / Pr_z
        
        if (Ratio_x / Ratio_z) > gamma:
            C += 1
        
    return C / len(Z)

# def membership_inference_attack(model, dataset, reference_models, k, a, gamma, online=True, batch_size=1):
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     Z = dataset.imgs[:len(dataset) // 10]  # Choose subset Z
#     C = 0
    
#     if online:
#         Theta_in = set()
#         for _ in range(k):
#             Di = np.random.choice(dataset.imgs, size=len(dataset) // 10, replace=False)
#             theta_x = train_model(Di.tolist() + [dataset.imgs[0]])  # Train model including x
#             Theta_in.add(theta_x)
#         Pr_x = sum(Pr(dataset.imgs[0], theta_prime) for theta_prime in Theta_in) / (2 * k) + sum(Pr(dataset.imgs[0], theta_prime) for theta_prime in reference_models) / (2 * k)
#     else:
#         Pr_x_OUT = sum(Pr(dataset.imgs[0], theta_prime) for theta_prime in reference_models) / k
#         Pr_x = 0.5 * ((1 + a) * Pr_x_OUT + (1 - a))
    
#     Ratio_x = Pr(dataset.imgs[0], model) / Pr_x
    
#     for z in Z:
#         Pr_z = sum(Pr(z, theta_prime) for theta_prime in reference_models) / k
#         Ratio_z = Pr(z, model) / Pr_z
        
#         if (Ratio_x / Ratio_z) > gamma:
#             C += 1
    
#     return C / len(Z)

# def Pr(sample, model):
#     with torch.no_grad():
#         sample = torch.tensor(sample).unsqueeze(0).to(model.device)
#         output = model(sample)
#     return torch.softmax(output, dim=1).max().item()

# def train_model(dataset):
#     model = nn.Sequential(
#         nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Flatten(),
#         nn.Linear(16 * 32 * 32, 44)
#     )
#     model.to("cuda" if torch.cuda.is_available() else "cpu")
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
    
#     for epoch in range(5):
#         for img, label in dataset:
#             img, label = img.to(model.device), torch.tensor(label).to(model.device)
#             optimizer.zero_grad()
#             output = model(img.unsqueeze(0))
#             loss = criterion(output, label.unsqueeze(0))
#             loss.backward()
#             optimizer.step()
    
#     return model