import os
import time
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from model import Generator, Discriminator
from utils import Anomaly_score, image_check

def main():
    epoch = 200
    batch_size = 1024
    learning_rate = 0.0002
    num_gpus = 1
    if num_gpus != 0:
        device = 'cuda'
    else:
        device = 'cpu'
    
    mnist_train = dset.MNIST("./", train=True, 
                            transform=transforms.Compose([
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                
                            ]),
                            target_transform=None,
                            download=True)
    mask = mnist_train.targets == 4
    mnist_train.data = mnist_train.data[mask]
    mnist_train.targets = mnist_train.targets[mask]


    mnist_test = dset.MNIST("./", train=False, 
                            transform=transforms.Compose([
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]),
                            target_transform=None,
                            download=True)

    train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)

    generator = nn.DataParallel(Generator(), device_ids=[0])
    discriminator = nn.DataParallel(Discriminator(), device_ids=[0])
    # generator = Generator().to(device)
    # discriminator = Discriminator().to(device)
    # loss function, optimizers, and labels for training

    loss_func = nn.BCELoss()
    gen_optim = torch.optim.Adam(generator.parameters(), lr= 5*learning_rate,betas=(0.5,0.999))
    dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,betas=(0.5,0.999))

    ones_label = Variable(torch.ones(batch_size,1)).to(device)
    zeros_label = Variable(torch.zeros(batch_size,1)).to(device)

    # train
    # print(mnist_train.data[0].shape)
    start = time.time()
    for i in range(epoch):
        print('Epoch: ', i, 'time(s): ', time.time()-start)
        gen_losses = []
        dis_losses = []

        for j,(image,label) in enumerate(train_loader):
            image = Variable(image).to(device)
            
            # generator
            gen_optim.zero_grad()
            
            z = Variable(init.normal_(torch.Tensor(batch_size,100),mean=0,std=0.1)).to(device)
            gen_fake = generator.forward(z)
            dis_fake,_ = discriminator.forward(gen_fake)
            
            gen_loss = torch.sum(loss_func(dis_fake,ones_label)) # fake classified as real
            gen_losses.append(gen_loss)
            gen_loss.backward(retain_graph=True)
            gen_optim.step()
        
            # discriminator
            dis_optim.zero_grad()
            
            z = Variable(init.normal_(torch.Tensor(batch_size,100),mean=0,std=0.1)).to(device)
            gen_fake = generator.forward(z)
            dis_fake,_ = discriminator.forward(gen_fake)
            
            dis_real,_ = discriminator.forward(image)
            dis_loss = torch.sum(loss_func(dis_fake,zeros_label)) + torch.sum(loss_func(dis_real,ones_label))
            dis_losses.append(dis_loss)
            dis_loss.backward()
            dis_optim.step()
        
            # model save
        if (i+1) % 1 == 0:
            # print('gen loss: ', gen_loss.item(), 'dis loss:', dis_loss.item())
            torch.save(generator.state_dict(),'./saved_model/generator_epoch{}.pkl'.format(i))
            torch.save(discriminator.state_dict(),'./saved_model/discriminator_epoch{}.pkl'.format(i))


            print("{}th iteration gen_loss: {} dis_loss: {}".format(i,gen_loss.data,dis_loss.data))
            v_utils.save_image(gen_fake.data[0:25],"./result/gen_epoch{}.png".format(i+1), nrow=5)
                
        # image_check(gen_fake.cpu())

    image_check(gen_fake.cpu())

    
    # start_idx = 64
    test_size = batch_size

    # test_data_mnist = mnist_test.__dict__['test_data'][start_idx:start_idx+batch_size]
    # test_data_mnist = test_data_mnist.view(test_size,1,28,28).type_as(torch.FloatTensor())
    # test_data_mnist.size()

    test_dataloader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)


    generator = nn.DataParallel(Generator(),device_ids=[0])
    discriminator = nn.DataParallel(Discriminator(),device_ids=[0])

    for ep, (test_data_mnist, label) in enumerate(test_dataloader):
        #generator,discriminator
        # z = Variable(init.normal(torch.zeros(test_size,100),mean=0,std=0.1),requires_grad=True)
        print('------ Start Anomaly Detection --------- data number ', ep)
        test_data_mnist = test_data_mnist.to(device)
        if label == 0:
            print('label == 0.....contnue')
            continue
        z = Variable(torch.randn(batch_size, 100, requires_grad=True, device=device))

        z_optimizer = torch.optim.Adam([z],lr=1)

        gen_fake = generator(z)
        loss = Anomaly_score(Variable(test_data_mnist), gen_fake, discriminator)
        print(loss.item())


        for i in range(500):
            print('updated noize z: ', z)
            gen_fake = generator(z)
            loss = Anomaly_score(Variable(test_data_mnist), gen_fake, discriminator, Lambda=0.01)
            loss.backward()
            z_optimizer.step()
            
            if i%10==0:
                print('update count: ', i, 'anomaly score: ', loss.cpu().item(), 'label: ', label.item(), 'time(s): ', time.time()-start)
                # '''
                # target = test_data_mnist[1,0,:,:].numpy()
                # plt.imshow(target,cmap="gray")
                # plt.show()
                
                # img=gen_fake.cpu().data[1,0,:,:].numpy()
                # plt.imshow(img,cmap='gray')
                # plt.show()
                # '''

        # for idx in range(test_size):
        #     target = test_data_mnist.numpy()
        #     plt.imshow(target,cmap="gray")
        #     plt.show()
        #     print("real data")

        #     img=gen_fake.cpu().data[idx,0,:,:].numpy()
        #     plt.imshow(img,cmap='gray')
        #     plt.show()
        #     print("generated data")
        #     print("\n------------------------------------\n")

        break

    train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=1,shuffle=True,drop_last=True)
    for ep, (test_data_mnist, label) in enumerate(train_loader):
        #generator,discriminator
        # z = Variable(init.normal(torch.zeros(test_size,100),mean=0,std=0.1),requires_grad=True)
        print('------ Start Anomaly Detection --------- data number ', ep)
        test_data_mnist = test_data_mnist.to(device)
        z = Variable(torch.randn(batch_size, 100, requires_grad=True, device=device))

        z_optimizer = torch.optim.Adam([z],lr=1)

        gen_fake = generator(z)
        loss = Anomaly_score(Variable(test_data_mnist), gen_fake, discriminator)
        print(loss.item())


        for i in range(500):
            print('updated noize z: ', z.item())
            gen_fake = generator(z)

            loss = Anomaly_score(Variable(test_data_mnist), gen_fake, discriminator, Lambda=0.01)
            loss.backward()
            z_optimizer.step()
            
            if i%10==0:
                print('update count: ', i, 'anomaly score: ', loss.cpu(), 'label: ', label.item(), 'time(s): ', time.time()-start)
                # '''
                # target = test_data_mnist[1,0,:,:].numpy()
                # plt.imshow(target,cmap="gray")
                # plt.show()
                
                # img=gen_fake.cpu().data[1,0,:,:].numpy()
                # plt.imshow(img,cmap='gray')
                # plt.show()
                # '''

        # for idx in range(test_size):
        #     target = test_data_mnist.numpy()
        #     plt.imshow(target,cmap="gray")
        #     plt.show()
        #     print("real data")

        #     img=gen_fake.cpu().data[idx,0,:,:].numpy()
        #     plt.imshow(img,cmap='gray')
        #     plt.show()
        #     print("generated data")
        #     print("\n------------------------------------\n")

        break

if __name__ == '__main__':
    main()