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


import argparse
from ast import arg, parse

parser = argparse.ArgumentParser(description='Metric_Learning_for_Anomaly_Detection')

# learning params
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--anomaly_setting', type=int, default=0)
parser.add_argument('--num_devices', type=int, default=1)
parser.add_argument('--data_num', type=int, default=20000)

args = parser.parse_args()
print(args)
print()

def main():
    epoch = args.max_epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    num_gpus = 1
    if num_gpus != 0:
        device = 'cuda'
        devices = [i for i in range(args.num_devices)]
    else:
        device = 'cpu'
    
    mnist_train = dset.MNIST("./", train=True, 
                            transform=transforms.Compose([
                                transforms.Resize(28),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                            ]),
                            target_transform=None,
                            download=True)

    # anomaly label setting
    if args.anomaly_setting == 0:
        normal_mask = (mnist_train.targets == 4)
    elif args.anomaly_setting == 1:
        normal_mask = (mnist_train.targets % 2 == 0)
    elif args.anomaly_setting == 2:
        normal_mask = (mnist_train.targets != 9)
    else:
        normal_mask = (mnist_train.targets >= 0)   

    if len(normal_mask[normal_mask]) > args.data_num:
        print("normal_mask: {} args.data_num: {}".format(len(normal_mask[normal_mask]), args.data_num))
        data_num = len(normal_mask[normal_mask])
    else :
        data_num = args.data_num

    mnist_train.data = mnist_train.data[normal_mask][:data_num]
    mnist_train.targets = mnist_train.targets[normal_mask][:data_num]


    mnist_test = dset.MNIST("./", train=False, 
                            transform=transforms.Compose([
                                transforms.Resize(28),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                            ]),
                            target_transform=None,
                            download=True)
    
    # anomaly label setting
    if args.anomaly_setting == 0:
        anomaly_mask = (mnist_test.targets != 4)
    elif args.anomaly_setting == 1:
        anomaly_mask = (mnist_test.targets % 2 != 0)
    elif args.anomaly_setting == 2:
        anomaly_mask = (mnist_test.targets == 9)
    else:
        anomaly_mask = (mnist_test.targets >= 0) 

    mnist_test.data = mnist_test.data[anomaly_mask][:data_num]
    mnist_test.targets = mnist_test.targets[anomaly_mask][:data_num]

    train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)

    generator = nn.DataParallel(Generator(), device_ids=devices).to(device)
    discriminator = nn.DataParallel(Discriminator(), device_ids=devices).to(device)
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
        print('Epoch:{} time(s):{:.2f}'.format(i, time.time()-start))
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
        
            # discriminator
            dis_optim.zero_grad()
            
            # z = Variable(init.normal_(torch.Tensor(batch_size,100),mean=0,std=0.1)).to(device)
            # gen_fake = generator.forward(z)
            # dis_fake,_ = discriminator.forward(gen_fake)
            
            dis_real,_ = discriminator.forward(image)
            dis_loss = torch.sum(loss_func(dis_fake,zeros_label)) + torch.sum(loss_func(dis_real,ones_label))
            dis_losses.append(dis_loss)
            dis_loss.backward()
            
            gen_optim.step()
            dis_optim.step()
            
        
            # model save
        if (i+1) % 1 == 0:
            # print('gen loss: ', gen_loss.item(), 'dis loss:', dis_loss.item())
            torch.save(generator.state_dict(),'./saved_model/anomaly_setting_{}/generator_epoch{}.pkl'.format(args.anomaly_setting, i))
            torch.save(discriminator.state_dict(),'./saved_model/anomaly_setting_{}/discriminator_epoch{}.pkl'.format(args.anomaly_setting, i))


            print("Epoch:{} gen_loss: {:.4f} dis_loss: {:.4f} time(s):{:.2f} ".format(i, gen_loss.data, dis_loss.data, time.time()-start))
            v_utils.save_image(gen_fake.data[0:25],"./result/anomaly_setting_{}/gen_epoch{}.png".format(args.anomaly_setting, i+1), nrow=5)
                
        # image_check(gen_fake.cpu())

    # image_check(gen_fake.cpu())

    
    # start_idx = 64
    test_size = batch_size

    # test_data_mnist = mnist_test.__dict__['test_data'][start_idx:start_idx+batch_size]
    # test_data_mnist = test_data_mnist.view(test_size,1,28,28).type_as(torch.FloatTensor())
    # test_data_mnist.size()

    test_dataloader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)


    generator = nn.DataParallel(Generator(),device_ids=devices).to(device)
    discriminator = nn.DataParallel(Discriminator(),device_ids=devices).to(device)

    for ep, (test_data_mnist, label) in enumerate(test_dataloader):
        #generator,discriminator
        # z = Variable(init.normal(torch.zeros(test_size,100),mean=0,std=0.1),requires_grad=True)
        print('------ Start Anomaly Detection --------- data number ', ep)
        test_data_mnist = test_data_mnist.to(device)
        # if label == 4:
        #     print('label == 4.....contnue')
        #     continue
        z = torch.randn(batch_size, 100, requires_grad=True, device=device)

        z_optimizer = torch.optim.Adam([z],lr=1)

        gen_fake = generator(z)
        loss = Anomaly_score(Variable(test_data_mnist), gen_fake, discriminator)
        # print(loss.item())


        for i in range(500):
            # print('updated noize z: ', z)
            gen_fake = generator(z)
            loss = Anomaly_score(Variable(test_data_mnist), gen_fake, discriminator, Lambda=0.01)
            loss.backward()
            z_optimizer.step()
            
            if i%10==0:
                print('Test update count:{} AS:{:.4f} label:{} time(s):{:.2f} '.format(i, loss.cpu().item(), label.item(), time.time()-start))
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
        z = torch.randn(batch_size, 100, requires_grad=True, device=device)

        z_optimizer = torch.optim.Adam([z],lr=1)

        gen_fake = generator(z)
        loss = Anomaly_score(Variable(test_data_mnist), gen_fake, discriminator)
        # print(loss.item())


        for i in range(500):
            # print('updated noize z: ', z.item())
            gen_fake = generator(z)

            loss = Anomaly_score(Variable(test_data_mnist), gen_fake, discriminator, Lambda=0.01)
            loss.backward()
            z_optimizer.step()
            
            if i%10==0:
                print('update count:{} AS:{:.4f} label:{} time(s):{:.2f} '.format(i, loss.cpu().item(), label.item(), time.time()-start))
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