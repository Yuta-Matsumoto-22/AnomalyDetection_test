import os
from unittest import result
from statistics import mean

import torch
import torch.nn as nn
from torch import optim

from utils import device_setting, seed_torch
from data_manager import AnomalyDataManager
from model import MLP, AutoEncoder
from metric_layer import ArcMarginProduct, MetricModel
from trainer import AutoEncoderTrainer

import argparse
from ast import arg, parse

parser = argparse.ArgumentParser(description='Metric_Learning_for_Anomaly_Detection')
# parser.add_argument('--', type=, default=)

# directory params
parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--model_dir', type=str, default='../models')
parser.add_argument('--result_dir', type=str, default='../results')

# settings
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_interval', type=int, default=1)

# learning params
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--criterion', type=str, default='MAE')
# parser.add_argument('--momentum', type=float, default=0.9)

# dataset
parser.add_argument('--dataset', type=str, default='kdd')
# parser.add_argument('--data_type', type=str, default='table')
parser.add_argument('--anormaly_label', type=int, default=4)
parser.add_argument('--data_num', type=int, default=200)
parser.add_argument('--only_normal', type=bool, default=True)

# model's params
parser.add_argument('--metric_layer', type=str, default='arcface')
parser.add_argument('--model', type=str, default='autoencoder')
parser.add_argument('--feature_dim', type=int, default=5)

args = parser.parse_args()
print(args)
print()

def main(args):
    # device, seed setting
    device = device_setting(gpu=args.gpu)
    seed_torch(args.seed)

    # prepare dataset
    data_manager = AnomalyDataManager(dataset=args.dataset, data_dir=args.data_dir, trans=None, seed=args.seed, only_normal=args.only_normal, anomaly_label=args.anormaly_label, data_num=200)
    
    # build dataloader
    dataloader_dict = data_manager.build_dataloader(args.batch_size)

    # build model, optimizer
    channel_in = data_manager.get_channel_in()
    num_classes = data_manager.get_num_classes()

    if data_manager.data_type.lower() == 'image':
        model = ResNet18(num_classes=args.feature_dim, channel_in=channel_in)
    elif data_manager.data_type.lower() == 'table':
        if args.model.lower() == 'mlp':
            model = MLP(in_features=data_manager.input_dim, out_features=args.feature_dim)
        elif args.model.lower() == 'autoencoder':
            model = AutoEncoder(in_dims=data_manager.input_dim, latent_dims=args.feature_dim, first_dims=32)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.criterion.lower() == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion.lower() == 'mae':
        criterion = nn.L1Loss()

    # dir setting
    model_dir = os.path.join(args.model_dir, args.dataset)
    model_dir = os.path.join(args.model_dir, "emb_size_" + str(args.feature_dim))
    if os.path.exists(model_dir) != True:
        os.makedirs(model_dir)

    result_dir = os.path.join(args.result_dir, args.dataset)
    result_dir = os.path.join(args.result_dir, "emb_size_" + str(args.feature_dim))
    if os.path.exists(result_dir) != True:
        os.makedirs(result_dir)

    # train model
    trainer = AutoEncoderTrainer(model, optimizer, criterion, device, model_dir=model_dir, result_dir=result_dir)
    trainer.train_model(dataloader_dict=dataloader_dict, max_epoch=args.max_epoch, save_interval=args.save_interval)

    # eval model on train dataset
    acc = 0
    model.eval()
    model.to(device)
    if args.criterion.lower() == 'mse':
        criterion = nn.MSELoss(reduction='none')
    elif args.criterion.lower() == 'mae':
        criterion = nn.L1Loss(reduction='none')

    model.set_threshold(dataloader_dict['train'], criterion, device, thres='mean')

    losses = []
    for data, label in dataloader_dict['train']: 
        data = data.to(device)
        label = label.to(device)
        pred_label, loss = model.inference(data, criterion, device)
        losses.append(mean(loss.cpu().detach().tolist()))
        acc += torch.sum(pred_label == label).item()

    acc = acc / len(dataloader_dict['train'].dataset)
    # plot, print result
    print('Train accuracy: {:4f} Train Loss: {:4f} '.format(acc, mean(losses)))

    # eval model on test dataset
    normal_acc = 0
    for data, label in dataloader_dict['test']:
        data = data.to(device)
        label = label.to(device)
        # １以上はすべて異常データ
        label_mask = label != 0
        label[label_mask] = 1
        # label = label.to(device)
        pred_label, loss = model.inference(data, criterion, device)
        losses.append(mean(loss.cpu().detach().tolist()))
        normal_acc += torch.sum(pred_label == label).item()

    normal_acc = normal_acc / len(dataloader_dict['test'].dataset)
    # plot, print result
    print('Test Normal accuracy: {:4f} Test Loss: {:4f} '.format(normal_acc, mean(losses)))

    pass


if __name__ == '__main__':
    main(args)