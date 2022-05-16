import enum
import os
import time
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


class GanTrainer():
    def __init__(self, generator, discriminator, 
                z_dim, gen_optim, dis_optim, learning_epoch,
                train_dataset, test_dataset, batch_size, device):
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.gen_optim = gen_optim
        self.dis_optim = dis_optim
        self.learning_epoch = learning_epoch

        self.train_dataset = train_dataset
        self.test_dataet = test_dataset
        # if val_dataset != None:
        #     self.val_dataset = val_dataset

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.BCELoss()
        self.ones_label = Variable(torch.ones(batch_size,1)).to(device)
        self.zeros_label = Variable(torch.zeros(batch_size,1)).to(device)

        self.gen_loss_dict = {}
        self.dis_loss_dict = {}
        self.anomaly_scores = {}

        self.start_time = time.time()

    def train_models(self, epoch=None):
        if epoch == None:
            epoch = self.learning_epoch
        else :
            pass
        for ep in range(epoch):
            for iter, (image, label) in enumerate(self.train_dataloader):
                image = Variable(image).to(self.device)

        pass

    def train_generator(self):
        pass

    def train_discriminator(self):
        pass

    def test_models(self):
        pass
    
    def generate(self):
        pass

    def detect_anomaly(self):
        pass

    def show_gen_data(self):
        pass

    def save_models(self):
        pass

    def save_results(self):
        pass