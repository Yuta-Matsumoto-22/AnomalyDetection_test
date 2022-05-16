import os
from unittest import result

import torch
import torch.nn as nn
from torch import optim

from utils import device_setting, seed_torch
from data_manager import AnomalyDataManager
from model import ResNet50MetricModel,ResNet50, MLP
from metric_layer import ArcMarginProduct, MetricModel
from resnet import ResNet18
from trainer import MetricTrainer

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
parser.add_argument('--wd', type=float,default=5e-4)
# parser.add_argument('--momentum', type=float, default=0.9)

# dataset
parser.add_argument('--dataset', type=str, default='kdd')
parser.add_argument('--data_type', type=str, default='table')
parser.add_argument('--anormaly_label', type=int, default=4)
parser.add_argument('--data_num', type=int, default=200)

# model's params
parser.add_argument('--metric_layer', type=str, default='arcface')
parser.add_argument('--feature_dim', type=int, default=256)

args = parser.parse_args()
print(args)
print()

def main(args):
    # device, seed setting
    device = device_setting(gpu=args.gpu)
    seed_torch(args.seed)

    # prepare dataset
    data_manager = AnomalyDataManager(dataset=args.dataset, data_dir=args.data_dir, trans=None, seed=args.seed, only_normal=None, anomaly_label=args.anormaly_label, data_num=200)
    
    # build dataloader
    dataloader_dict = data_manager.build_dataloader(args.batch_size)

    # build model, optimizer
    channel_in = data_manager.get_channel_in()
    num_classes = data_manager.get_num_classes()

    if args.data_type.lower() == 'image':
        model = ResNet18(num_classes=args.feature_dim, channel_in=channel_in)
    elif args.data_type.lower() == 'table':
        model = MLP(in_features=data_manager.input_dim, out_features=args.feature_dim)
    metric_layer = ArcMarginProduct(in_features=args.feature_dim, out_features=num_classes, s=30.0, m=0.80, easy_margin=False)
    optimizer = optim.Adam(list(model.parameters()) + list(metric_layer.parameters()), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    metric_model = MetricModel(model, metric_layer, num_classes)

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
    trainer = MetricTrainer(model, metric_layer, optimizer, criterion, device, model_dir=model_dir, result_dir=result_dir)
    trainer.train_model(dataloader_dict=dataloader_dict, max_epoch=args.max_epoch, save_interval=args.save_interval)

    # set center of gravity for classification
    metric_model.set_center_of_classes(dataloader_dict['train'], args.feature_dim, device)

    # eval model on train dataset
    acc = 0
    metric_model.eval()
    for data, label in dataloader_dict['train']: 
        data = data.to(device)
        # label = label.to(device)
        probs = metric_model.inference(data)
        _, pred_label = torch.max(probs, 1)
        acc += torch.sum(pred_label == label).item()

    acc = acc / len(dataloader_dict['train'].dataset)
    # plot, print result
    print('Train accuracy: {:4f}'.format(acc))

    # eval model on test dataset
    normal_acc = 0
    anomaly_detection_acc = 0
    for data, label in dataloader_dict['test']:
        data = data.to(device)
        # 学習にない異常データ（ラベル22~）はその他ラベル-1とする
        label_mask = label > 21
        label[label_mask] = -1
        # label = label.to(device)
        probs = metric_model.inference(data)
        pred_values, pred_label = torch.max(probs, 1)
        normal_acc += torch.sum(pred_label == label).item()

        anomaly_mask = pred_values < 0.5
        pred_label[anomaly_mask] = -1
        anomaly_detection_acc += torch.sum(pred_label == label).item()

    normal_acc = normal_acc / len(dataloader_dict['test'].dataset)
    anomaly_detection_acc = anomaly_detection_acc / len(dataloader_dict['test'].dataset)
    # plot, print result
    print('Test Normal accuracy: {:4f}'.format(normal_acc))
    print('Test Anomaly Detection accuracy: {:4f}'.format(anomaly_detection_acc))

    pass


if __name__ == '__main__':
    main(args)