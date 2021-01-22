# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import math
from random import Random

size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
# print('Successfully get size!')
rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
# print('Successfully get rank!')

# Required PyTorch Module 
import torch
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Models 
from cjltest.models import MnistCNN, AlexNetForCIFAR, LeNetForMNIST
import cjltest.utils_data
from cjltest.utils_model import MySGD, test_model
import ResNetOnCifar10
import VGGOnCifar10
import VGGOnCifar100

def get_data_transform(name):
    if name == 'mnist':
        return cjltest.utils_data.get_data_transform(name)
    if name == 'cifar':
        transform_train = transforms.Compose([ 
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
        ])
        transform_test = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
        ])
        return transform_train, transform_test 

class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=None, seed=1234):
        if sizes is None:
            sizes = [0.7, 0.2, 0.1]
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for ratio in sizes:
            part_len = int(ratio * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(dataset, workers, iid, p=(0.6, 0.4)):
    """ Partitioning Data """
    workers_num = len(workers)
    if iid:
        partition_sizes = [1.0 / workers_num for _ in range(workers_num)]
    else:
        partition_sizes = [float((i+1) * 2 / ((1+workers_num) * workers_num)) * p[0] for i in range(workers_num)] + [p[1]]
    partition = DataPartitioner(dataset, partition_sizes)
    return partition


def select_dataset(workers: list, rank: int, partition: DataPartitioner, batch_size: int):
    workers_num = len(workers)
    partition_dict = {workers[i]: i for i in range(workers_num)}
    partition = partition.use(partition_dict[rank])
    return DataLoader(partition, batch_size=batch_size, shuffle=True)

# Parameter Server and Learner 
import param_server
import learner

import argparse
parser = argparse.ArgumentParser()
# Training info
parser.add_argument('--title', type=str, default='FedNova')

# model and datasets 
parser.add_argument('--data-dir', type=str, default='~/dataset')
parser.add_argument('--model', type=str, default='ResNet18OnCifar10')
parser.add_argument('--save-path', type=str, default='./')
parser.add_argument('--num-gpu', type=int, default=1)

# Hyper parameters setting 
parser.add_argument('--iid', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--train-bsz', type=int, default=32) # Individual batch size
parser.add_argument('--local-iteration', type=str, default='linear')

args = parser.parse_args()

# print('Begin get models!')
""" Get model and train/test datasets """
if args.model == 'MnistCNN':
    model = MnistCNN()
    train_transform, test_transform = get_data_transform('mnist')
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, download=False, transform=test_transform)

elif args.model == 'LeNet':
    model = LeNetForMNIST()
    train_transform, test_transform = get_data_transform('mnist')
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, download=False, transform=test_transform)

elif args.model == 'LROnMnist':
    model = ResNetOnCifar10.LROnMnist()
    train_transform, test_transform = get_data_transform('mnist')
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, download=False, transform=test_transform)

elif args.model == 'LROnFashionMnist':
    model = ResNetOnCifar10.LROnMnist()
    train_transform, test_transform = get_data_transform('mnist')
    train_dataset = datasets.FashionMNIST(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.FashionMNIST(args.data_dir, train=False, download=False, transform=test_transform)

elif args.model == 'LROnCifar10':
    model = ResNetOnCifar10.LROnCifar10()
    train_transform, test_transform = get_data_transform('cifar')
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=test_transform)

elif args.model == 'AlexNetOnCifar10':
    model = AlexNetForCIFAR()
    train_transform, test_transform = get_data_transform('cifar')
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=test_transform)

elif args.model == 'AlexNetOnCifar100':
    model = AlexNetForCIFAR(num_classes=100)
    train_transform, test_transform = get_data_transform('cifar')
    train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=False, transform=test_transform)
        
elif args.model == 'ResNet18OnCifar10':
    model = ResNetOnCifar10.ResNet18()
    train_transform, test_transform = get_data_transform('cifar')
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=test_transform)

elif args.model == 'VGG11OnCifar10':
    model = VGGOnCifar10.vgg11()
    train_transform, test_transform = get_data_transform('cifar')
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=test_transform)

elif args.model == 'VGG16OnCifar10':
    model = VGGOnCifar10.vgg16()
    train_transform, test_transform = get_data_transform('cifar')
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=test_transform)

elif args.model == 'VGG11OnCifar100':
    model = VGGOnCifar100.vgg11()
    train_transform, test_transform = get_data_transform('cifar')
    train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=False, transform=test_transform)

else:
    print('Model is not found!')
    sys.exit(-1)

"""  Start and Run  """
if rank == 0:
    # This is parameter server 
    iterationPerEpoch = math.ceil(len(train_dataset) / ((size - 1) * args.train_bsz))
    test_data = DataLoader(test_dataset, batch_size=400, shuffle=False)
    param_server.init_processes(rank, size, model, args, iterationPerEpoch, test_data)
else:
    # This is worker nodes
    workers = [v+1 for v in range(size-1)]
    train_bsz, test_bsz = args.train_bsz, 400
    train_data = select_dataset(workers, rank, partition_dataset(train_dataset, workers, args.iid), batch_size=train_bsz)
    test_data  = select_dataset(workers, rank, partition_dataset(test_dataset, workers, args.iid),  batch_size=test_bsz)
    weight = (1/(size-1)) if args.iid else (float(rank * 2 / ((size - 1) * size)))
    learner.init_processes(rank, size, model, args, train_data, test_data, weight)

print('Rank {} finishes. '.format(rank))

