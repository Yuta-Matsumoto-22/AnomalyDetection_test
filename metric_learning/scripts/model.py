from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from metric_layer import ArcMarginProduct, AddMarginProduct, SphereProduct

class DCGAN_Discriminator(nn.Module):
    def __init__(self, channel_in, ndf=64):
        super(DCGAN_Discriminator, self).__init__()
        self.fe = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channel_in, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_layer = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # out = self.layer1(x)
        # out = self.layer2(out)
        # # out = out.view(out.size()[0], -1)
        # feature = out
        # out = self.out_layer(out)
        # out = out.view(-1, 1)
        feature = self.fe(x)
        out = self.out_layer(feature)
        return out.view(-1, 1), feature

class ResNet50(nn.Module):
    def __init__(self, channel_in, output_dim):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        # Block 1
        self.block0 = self._building_block(256, channel_in=64)
        self.block1 = nn.ModuleList([
            self._building_block(256) for _ in range(2)
        ])
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
        
        # Block 2
        self.block2 = nn.ModuleList([
            self._building_block(512) for _ in range(4)
        ])
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))
        
        # Block 3
        self.block3 = nn.ModuleList([
            self._building_block(1024) for _ in range(6)
        ])
        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2))
        
        # Block 4
        self.block4 = nn.ModuleList([
            self._building_block(2048) for _ in range(3)
        ])
        self.avg_pool = GlobalAvgPool2d()  # TODO: GlobalAvgPool2d

        # state (bs, 2048)
        self.fc = nn.Linear(2048, output_dim)
        # self.out = nn.Linear(1000, output_dim)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)

        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = h.view(-1, h.size(1))
        h = self.fc(h)
        # h = torch.relu(h)
        # h = self.out(h)
        # y = torch.log_softmax(h, dim=-1)
        return h

    def _building_block(self, channel_out, channel_in=None):
        if channel_in is None:
            channel_in = channel_out
        return Block(channel_in, channel_out)

class Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        channel = channel_out // 4
        # 1x1 の畳み込み
        self.conv1 = nn.Conv2d(channel_in, channel, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU()
        # 3x3 の畳み込み
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU()
        # 1x1 の畳み込み
        self.conv3 = nn.Conv2d(channel, channel_out, kernel_size=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(channel_out)
        # skip connection用のチャネル数調整        
        self.shortcut = self._shortcut(channel_in, channel_out)
        
        self.relu3 = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)  # skip connection
        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=(1, 1), padding=0)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class ResNet50MetricModel(nn.Module):
    def __init__(self, channel_in, feature_dim, output_dim):
        super(ResNet50MetricModel, self).__init__()
        self.output_dim = output_dim

        # Define ResNet50 model 
        self.conv1 = nn.Conv2d(channel_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        # Block 1
        self.block0 = self._building_block(256, channel_in=64)
        self.block1 = nn.ModuleList([
            self._building_block(256) for _ in range(2)
        ])
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
        
        # Block 2
        self.block2 = nn.ModuleList([
            self._building_block(512) for _ in range(4)
        ])
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))
        
        # Block 3
        self.block3 = nn.ModuleList([
            self._building_block(1024) for _ in range(6)
        ])
        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2))
        
        # Block 4
        self.block4 = nn.ModuleList([
            self._building_block(2048) for _ in range(3)
        ])
        self.avg_pool = GlobalAvgPool2d()  # TODO: GlobalAvgPool2d

        self.metric = ArcMarginProduct(in_features=feature_dim, out_features=output_dim)

        # FC Layer
        self.fc = nn.Linear(2048, feature_dim)
        self.out = nn.Linear(1000, output_dim)

    def forward(self, x, label):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)

        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = h.view(-1, h.size(1))

        emb = self.fc(h)
        # emb = torch.relu(h)
        # emb = self.out(h)
        h = self.metric(emb, label)
        
        # h = torch.log_softmax(h, dim=-1)

        return h, emb

    def inference(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)

        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = h.view(-1, h.size(1))

        emb = self.fc(h)
        # h = torch.relu(h)
        # h = self.metric(emb, label)

        return emb

    def _building_block(self, channel_out, channel_in=None):
        if channel_in is None:
            channel_in = channel_out

        return Block(channel_in, channel_out)

    def set_center_of_classes(self, train_dataloader):
        self.centers = [[] for _ in range(len(self.output_dim))]
        for data, label in train_dataloader:
            out = self.inference(data)
            self.centers[label].append(out)


class ResNet18MetricModel(nn.Module):
    def __init__(self, channel_in, feature_dim, output_dim):
        super(ResNet50MetricModel, self).__init__()
        self.output_dim = output_dim

        # Define ResNet50 model 
        self.conv1 = nn.Conv2d(channel_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        # Block 1
        self.block0 = self._building_block(256, channel_in=64)
        self.block1 = nn.ModuleList([
            self._building_block(256) for _ in range(2)
        ])
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
        
        # Block 2
        self.block2 = nn.ModuleList([
            self._building_block(512) for _ in range(4)
        ])
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))
        
        # Block 3
        self.block3 = nn.ModuleList([
            self._building_block(1024) for _ in range(6)
        ])
        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2))
        
        # Block 4
        self.block4 = nn.ModuleList([
            self._building_block(2048) for _ in range(3)
        ])
        self.avg_pool = GlobalAvgPool2d()  # TODO: GlobalAvgPool2d

        self.metric = ArcMarginProduct(in_features=feature_dim, out_features=output_dim)

        # FC Layer
        self.fc = nn.Linear(2048, feature_dim)
        self.out = nn.Linear(1000, output_dim)

    def forward(self, x, label):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)

        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = h.view(-1, h.size(1))

        emb = self.fc(h)
        # emb = torch.relu(h)
        # emb = self.out(h)
        h = self.metric(emb, label)
        
        # h = torch.log_softmax(h, dim=-1)

        return h, emb

    def inference(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)

        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = h.view(-1, h.size(1))

        emb = self.fc(h)
        # h = torch.relu(h)
        # h = self.metric(emb, label)

        return emb

    def _building_block(self, channel_out, channel_in=None):
        if channel_in is None:
            channel_in = channel_out

        return Block(channel_in, channel_out)

    def set_center_of_classes(self, train_dataloader):
        self.centers = [[] for _ in range(len(self.output_dim))]
        for data, label in train_dataloader:
            out = self.inference(data)
            self.centers[label].append(out)

        


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
