import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

dset = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype = torch.float).scatter_(dim = 0, index = torch.tensor(y), value = 1))
)

#ToTensor converts PIL images/numpy arrays to tensors with a float dtype
#features in FashionMNIST are PIL images with int labels
#we need tensors and one-hot encoded tensors


