import numpy as np 
import matplotlib.pyplot as plt 
import torch
from tqdm.notebook import tqdm
from torchvision import datasets,transforms

mnist_train = datasets.MNIST(root = "./datasets" , train = True , transform = transforms.ToTensor ,download = True)
mnist_test = datasets.MNIST(root = "./datasets" , train = False , transform = transforms.ToTensor ,download = True)

print(f'number of MNIST training examples : {format(len(mnist_train))}')
print(f'number of MNIST test examples : {format(len(mnist_test))}')

image , label = mnist_train[3] 