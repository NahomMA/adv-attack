from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch
from torch import nn


class MNISTDataLoader:
  def __init__(self, batch_size,num_workers):
    # transform train
    self.train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])


    #train  data and data loader 
    self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=self.train_transform)   

    # train val data loader
    self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  


    #transform test
    self.test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # test data
    self.test_data = datasets.MNIST(root='./data', train=False, download=True, transform=self.test_transform)
    self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  def get_data_loaders(self):
    return self.train_loader,  self.test_loader
