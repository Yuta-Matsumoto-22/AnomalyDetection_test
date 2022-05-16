import os

import torch

class MetricTrainer():
    def __init__(self, model, metric_layer, optimizer, criterion, device, model_dir, result_dir):
        self.model = model.to(device)
        self.metric_layer = metric_layer.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_dir = model_dir
        self.result_dir = result_dir
        self.device = device

    def train_model(self, dataloader_dict, max_epoch, save_interval=10, save_flag=True):

        self.model.train()

        print('------------ Training ------------')
        for epoch in range(max_epoch):
            # print('------------ Epoch:{} ------------'.format(epoch))

            for i, (data, label) in enumerate(dataloader_dict['train']):
                data = data.to(self.device)
                label = label.to(self.device)

                
                feature = self.model(data)
                output = self.metric_layer(feature, label)
                loss = self.criterion(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            
            loss_dict, acc_dict = self._Evaluation_model(dataloader_dict)

            # print('Epoch:{}  Loss:{:.4f}   Acc:{:.4f}'.format(epoch, loss_dict['train'], acc_dict['train']))
            print('--------- Epoch {} ---------'.format(epoch))
            print(' Train Loss:{:.4f}  Acc.:{:.4f}'.format(loss_dict['train'], acc_dict['train']))

            if save_flag == True and (epoch % save_interval) == 0:
                model_path = os.path.join(self.model_dir, 'model_epoch_{}.pth'.format(epoch))
                self._save_model(model_path) 
                # self._save_result(loss_dict, acc_dict, self.result_path)

    def eval_model(self, dataloader):
        self.model.eval()
        acc = 0
        for data, label in dataloader:
            data = data.to(self.device)
            label = label.to(self.device)

            feature = self.model(data)
            output = self.metric_layer(feature, label)

            loss = self.criterion(output, label)

            # pred_label = self.model(data)
            outputs = self.model(data)
            outputs = self.metric_layer(outputs, label)
            _, pred_label = torch.max(outputs, 1)
            # print(pred_label)
            # print(label)

            # print(pred_label)
            # print(output)
            acc += torch.sum(pred_label == label).item()

        acc = acc / len(dataloader.dataset)

        return loss, acc

    def _Evaluation_model(self, dataloader_dict):
        loss_dict = {}
        acc_dict = {}

        # for key in dataloader_dict.keys():
        #     loss = self.eval_model(dataloader_dict[key])
        #     loss_dict[key] = loss
        #     # acc_dict[key] = acc
        loss, acc = self.eval_model(dataloader_dict['train'])
        loss_dict['train'] = loss
        acc_dict['train'] = acc

        return loss_dict, acc_dict
    
    def _save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        # print(file_path)
        # file_path = file_path.replace('model', 'metric_layer')
        # torch.save(self.metric_layer.state_dict(), file_path)

    # def _save_result(self, result, file_path):
        