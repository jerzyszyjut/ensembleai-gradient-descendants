from my_datasets import MembershipDataset, inference_dataloader
from load_my_model import load_model
import torch
from torchvision import transforms


BATCH_SIZE = 1
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]),
])

if __name__ == "__main__":
    model = load_model(model_name="resnet18", model_path="01_MIA_69.pt", device=DEVICE)      
    dataset: MembershipDataset = torch.load("dataset_1.pt", weights_only=False)
    dataset.transform = data_transforms
    dataloader = inference_dataloader(dataset, BATCH_SIZE)

    outputs_list = []

    correct = 0
    total = 0

    for id, img, label, membership in dataloader:
        print(label)
        img = img.to(DEVICE)
        label = label.to(DEVICE)

        with torch.no_grad():
            membership_output = model(img)
            print(membership_output)
            _, predicted = torch.max(membership_output, 1)
            print(predicted)

        outputs_list += membership_output.tolist()

        total += label.size(0)
        correct += (predicted == label).sum().item()
        break

    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy:.2f}%')