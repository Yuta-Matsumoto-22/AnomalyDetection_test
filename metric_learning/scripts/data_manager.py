import os
import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from collections import OrderedDict


class AnomalyDataManager():
    def __init__(self, dataset, data_dir, trans, seed, only_normal, anomaly_label=None, data_num=-1):
        self.dataset = dataset
        self.data_dir = data_dir
        self.transforms = trans
        self.anomaly_label = anomaly_label
        self.data_num = data_num
        self.seed = seed
        self.only_normal = only_normal

        self.dataset_dict = {}
        self.dataloader_dict = {}

        if self.dataset.lower() == 'mnist':
            self.data_type = 'image'
            if self.transforms == None:
                self.train_dataset = dset.MNIST(self.data_dir, train=True, 
                                        transform=transforms.Compose([
                                            # transforms.CenterCrop(224),
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,)),
                                            
                                        ]),
                                        target_transform=None,
                                        download=True)
                self.test_dataset = dset.MNIST(self.data_dir, train=False, 
                                        transform=transforms.Compose([
                                            # transforms.CenterCrop(224),
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, ), (0.5, )),
                                        ]),
                                        target_transform=None,
                                        download=True)
            else:
                self.train_dataset = dset.MNIST(self.data_dir, train=True, 
                                        transform=self.transforms,
                                        target_transform=None,
                                        download=True)
                self.test_dataset = dset.MNIST(self.data_dir, train=False, 
                                        transform=self.transforms,
                                        target_transform=None,
                                        download=True)
            
            assert anomaly_label != None, print('assert: anomaly_label==None.')
            # anomaly label setting
            if self.anomaly_label >= 0:
                mask = (self.train_dataset.targets != self.anomaly_label)
                print(len(self.train_dataset.targets))
            else:
                mask = (self.train_dataset.targets % 2 == 0)
            self.train_dataset.data = self.train_dataset.data[mask][:self.data_num]
            self.train_dataset.targets = self.train_dataset.targets[mask][:self.data_num]
            self.train_dataset.classes = self.train_dataset.targets.unique().detach().numpy()
            print(sorted(self.train_dataset.targets.unique().detach().numpy()))
            print(self.train_dataset.classes)
            trans_dict = {sorted(self.train_dataset.targets.unique().detach().numpy())[k]:k for k in range(len(self.train_dataset.classes))}
            print(trans_dict)
            
            # trans labels (ラベルが連続になるように)
            print(len(self.train_dataset.targets))
            self.train_dataset.targets = torch.tensor(list(map(lambda x: trans_dict[x.item()], self.train_dataset.targets)))
            print(self.train_dataset.targets)

            self.val_dataset = copy.deepcopy(self.train_dataset)

            self.train_dataset.data, self.val_dataset.data, self.train_dataset.targets, self.val_dataset.targets = train_test_split(self.train_dataset.data, self.train_dataset.targets,  test_size=0.2, random_state=self.seed, stratify=self.train_dataset.targets)

        elif self.dataset.lower() == 'kdd':
            self.data_type = 'table'
            self.train_dataset, self.val_dataset, self.test_dataset = self.load_KDD(only_normal)

        self.dataset_dict['train'] = self.train_dataset
        self.dataset_dict['val'] = self.val_dataset
        self.dataset_dict['test'] = self.test_dataset

    def build_dataloader(self, batch_size):
        for key in self.dataset_dict.keys():
            if key == 'train':
                self.dataloader_dict[key] = torch.utils.data.DataLoader(dataset=self.dataset_dict[key], batch_size=batch_size, shuffle=True, drop_last=False)
            else :
                self.dataloader_dict[key] = torch.utils.data.DataLoader(dataset=self.dataset_dict[key], batch_size=batch_size, shuffle=False, drop_last=False)

        return self.dataloader_dict

    def get_dataset(self):
        return self.dataset_dict

    def get_num_classes(self):
        if self.dataset.lower() == 'mnist':
            num_classes = len(self.train_dataset.classes)
        elif self.dataset.lower() == 'kdd':
            num_classes = self.num_classes

        return num_classes

    def get_channel_in(self):
        if len(self.train_dataset[0]) < 3:
            channel_in = 1
        else:
            channel_in = self.train_dataset.data[0].size()[0]

        return channel_in

    def preprocess_NSL(self, df, columns):
        cols = dict(zip(range(len(columns)), columns))
        df.rename(columns=cols, inplace=True)
        df.head()

        # target_dict = dict(zip(df['target'].unique(), range(len(df['target'].unique()))))

        self.target_dict.update(self.test_only_target)
        # print(self.target_dict)
        data = pd.get_dummies(df.iloc[:, :-2])
        target = df.iloc[:, -2].map(self.target_dict)
        # target = df.iloc[:, -2].map(self.test_only_target)
        # print(self.target_dict)

        return data, target

    def load_KDD(self, only_normal):
        kdd_path = os.path.join(self.data_dir, 'NSL-KDD')
        # kdd_path = self.data_dir + '\\NSL-KDD'
        df_train = pd.read_csv(kdd_path+'\KDDTrain+.txt', header=None)
        df_test = pd.read_csv(kdd_path+'\KDDTest+.txt', header=None)
        columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
                    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'target', 'level'
        ]

        print(columns)
        dos = ['back','land','neptune','pod','smurf',
            'teardrop','mailbomb','processtable','udpstorm','apache2','worm']
        probe = ['satan','ipsweep','nmap','portsweep','mscan','saint']
        r2l = ['guess_passwd','ftp_write','imap','phf','multihop','warezmaster', 'warezclient', 'xlock','xsnoop','snmpguess',
            'snmpgetattack','httptunnel','sendmail', 'named', 'spy']
        u2r = ['buffer_overflow','loadmodule','rootkit','perl','sqlattack','xterm','ps']

        self.target_dict = dict(zip(sorted(df_train.iloc[:, -2].unique()), range(len(df_train.iloc[:, -2].unique()))))
        print(self.target_dict)
        self.test_only_target_dict = {}
        # print(df_test.iloc[:, -2].unique())
        test_only_target = set(df_test.iloc[:, -2].unique()) - set(self.target_dict.keys())
        print(sorted(test_only_target))
        self.test_only_target = dict(zip(sorted(test_only_target), range(len(self.target_dict.keys()), len(self.target_dict.keys()) + len(test_only_target))))

        # print(self.test_only_target)
        # for key in test_only_attacks_dict.keys():
        #     if key not in self.target_dict.keys():
        #         # print(key)
        #         self.test_only_target[key] = -1

        # print(self.test_only_target)

        train_data, train_target = self.preprocess_NSL(df_train, columns)
        test_data, test_target = self.preprocess_NSL(df_test, columns)

        print(train_target.unique())
        print(test_target.unique())
        self.fill_missing_columns(train_data, test_data)

        train_data, val_data, train_target, val_target = train_test_split(train_data, train_target,  test_size=0.2, random_state=self.seed, stratify=train_target)

        # カテゴリカル変数以外を標準化
        scaling_columns = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
                    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                    'num_shells', 'num_access_files', 'num_outbound_cmds', 'count', 'srv_count', 'serror_rate',
                    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]

        ### 通常の標準化###
        # trainと言うDataFrameにfit
        sc = StandardScaler().fit(train_data[scaling_columns])

        # 標準化したカラムのみ元のDataFrameに戻す
        scaled_train = pd.DataFrame(sc.transform(train_data[scaling_columns]), columns=scaling_columns, index=train_data.index)
        scaled_val = pd.DataFrame(sc.transform(val_data[scaling_columns]), columns=scaling_columns, index=val_data.index)
        scaled_test = pd.DataFrame(sc.transform(test_data[scaling_columns]), columns=scaling_columns, index=test_data.index)

        train_data.update(scaled_train)
        val_data.update(scaled_val)
        test_data.update(scaled_test)

        train_data = torch.tensor(train_data.values).to(torch.float32)
        train_target = torch.tensor(train_target.values)

        val_data = torch.tensor(val_data.values).to(torch.float32)
        val_target = torch.tensor(val_target.values)

        test_data = torch.tensor(test_data.values).to(torch.float32)
        test_target = torch.tensor(test_target.values)

        print('Train target unique : {}'.format(len(train_target.unique())))
        print('Test target unique : {}'.format(len(test_target.unique())))

        self.num_classes = len(train_target.unique())
        self.input_dim = len(train_data[0]) 

        return torch.utils.data.TensorDataset(train_data, train_target), torch.utils.data.TensorDataset(val_data, val_target), torch.utils.data.TensorDataset(test_data, test_target)

    def fill_missing_columns(self, df_a, df_b):
        columns_for_b = set(df_a.columns) - set(df_b.columns)
        for column in columns_for_b:
            df_b[column] = 0
        columns_for_a = set(df_b.columns) - set(df_a.columns)
        for column in columns_for_a:
            df_a[column] = 0