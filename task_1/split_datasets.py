from my_datasets import MembershipDataset, inference_dataloader
import torch

def split_dataset_by_membership(dataset: MembershipDataset):
    dataset_0 = MembershipDataset(transform=dataset.transform)
    dataset_1 = MembershipDataset(transform=dataset.transform)

    for idx in range(len(dataset)):
        id_, img, label, membership = dataset[idx]
        if membership == 0:
            dataset_0.ids.append(id_)
            dataset_0.imgs.append(img)
            dataset_0.labels.append(label)
            dataset_0.membership.append(membership)
        elif membership == 1:
            dataset_1.ids.append(id_)
            dataset_1.imgs.append(img)
            dataset_1.labels.append(label)
            dataset_1.membership.append(membership)

    return dataset_0, dataset_1

if __name__ == "__main__":
    dataset: MembershipDataset = torch.load("pub.pt", weights_only=False)
    new_dataset_0, new_dataset_1 = split_dataset_by_membership(dataset)
    print("length of new dataset 0: " + str(len(new_dataset_0)))
    print("length of new dataset 1: " + str(len(new_dataset_1)))
    torch.save(new_dataset_0, 'dataset_0.pt')
    torch.save(new_dataset_1, 'dataset_1.pt')
