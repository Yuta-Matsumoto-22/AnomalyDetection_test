from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from metric_layer import ArcMarginProduct, AddMarginProduct, SphereProduct
from statistics import mean

class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers=3):
        super(MLP, self).__init__()
        # self.first_layer = HiddenLayer(in_features, 2*in_features)
        self.fe = nn.Sequential(
            HiddenLayer(in_features, 2*in_features), 
            nn.ReLU(),
            *[HiddenLayer(2*in_features, 2*in_features) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(2*in_features, out_features)

    def forward(self, x):
        fe = self.fe(x)
        out = self.output_layer(fe)

        return out


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class AutoEncoder(nn.Module):
    def __init__(self, in_dims, latent_dims, first_dims=28*28//2):
        super(AutoEncoder, self).__init__()
        self.in_dims = in_dims
        self.encoder = nn.Sequential(
            nn.Linear(in_dims, first_dims),
            nn.ReLU(),
            nn.Linear(first_dims, first_dims // 2),
            nn.ReLU(),
            nn.Linear(first_dims // 2, latent_dims)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, first_dims // 2),
            nn.ReLU(),
            nn.Linear(first_dims // 2, first_dims),
            nn.ReLU(),
            nn.Linear(first_dims, in_dims)
        )

        self.threshold = None

    def forward(self, x):
        fe = self.encoder(x)
        x = self.decoder(fe)
        return x, fe

    def inference(self, x, criterion, device):
        if self.threshold == None:
            print('assert set threshold')
            exit()
        x_hat, feature = self.forward(x)
        losses = torch.mean(criterion(x_hat, x), 1)
        pred_label = (losses > self.threshold).int()

        return pred_label, losses

    def set_threshold(self, dataloader, criterion, device, thres='max'):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.threshold = 0
        max_tmp = 0
        mean_loss = []
        for data, label in dataloader:
            data = data.reshape(-1, self.in_dims)
            data = data.to(device)
            label = label.to(device)
            out, fe = self.forward(data)
            losses = torch.mean(criterion(out, data), 1)
            loss_list = losses.cpu().detach().tolist()
            mean_loss.append(mean(loss_list))
            max_tmp = max(loss_list)
            max_idx = loss_list.index(max_tmp)
            # print(max_tmp)
            if self.threshold < max_tmp:
                self.threshold = max_tmp
                max_data = data[max_idx]

        if thres == 'max':
            print('set Threshold : {}'.format(self.threshold))
            return
        elif thres == 'mean':
            self.threshold = mean(mean_loss)
            print('set Threshold : {}'.format(self.threshold))
        else :
            self.threshold = (self.threshold + mean(mean_loss)) / 2
            print('set Threshold : {}'.format(self.threshold))
        # print('max_data: ')
        # print(max_data)
        # print('reconstruct: ')
        # print(self.forward(max_data)[0].cpu().detach().numpy())
        
        pass
