import os
import numpy as np
import torch

class AutoEncoderTrainer():
    def __init__(self, model, optimizer, criterion, input_size, device, model_dir, result_dir):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_dir = model_dir
        self.result_dir = result_dir
        self.device = device
        self.input_size = input_size

    def train_model(self, dataloader_dict, max_epoch, save_interval=10, save_flag=True):

        self.model.train()

        print('------------ Training ------------')
        for epoch in range(max_epoch):
            # print('------------ Epoch:{} ------------'.format(epoch))

            for i, (data, label) in enumerate(dataloader_dict['train']):
                data = data.reshape(-1, self.input_size)
                data = data.to(self.device)
                # label = label.to(self.device)
                output, feature = self.model(data)
                loss = self.criterion(output, data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            
            # loss_dict, acc_dict = self._Evaluation_model(dataloader_dict)
            loss_dict = self._Evaluation_model(dataloader_dict)

            # print('Epoch:{}  Loss:{:.4f}   Acc:{:.4f}'.format(epoch, loss_dict['train'], acc_dict['train']))
            print('--------- Epoch {} ---------'.format(epoch))
            # print(' Train Loss:{:.4f}  Acc.:{:.4f}'.format(loss_dict['train'], acc_dict['train']))
            print(' Train Loss: {:.4f}'.format(loss_dict['train']))

            if save_flag == True and (epoch % save_interval) == 0:
                model_path = os.path.join(self.model_dir, 'model_epoch_{}.pth'.format(epoch))
                self._save_model(model_path) 
                # self._save_result(loss_dict, acc_dict, self.result_path)

    def eval_model(self, dataloader):
        self.model.eval()
        acc = 0
        losses = []
        for data, label in dataloader:
            data = data.reshape(-1, self.input_size)
            data = data.to(self.device)
            label = label.to(self.device)

            output, feature = self.model(data)
            loss = self.criterion(output, data)
            losses.append(loss.cpu().detach().numpy())

        #     acc += torch.sum(pred_label == label).item()

        # acc = acc / len(dataloader.dataset)

        return np.array(losses).mean()

    def _Evaluation_model(self, dataloader_dict):
        loss_dict = {}
        acc_dict = {}

        # for key in dataloader_dict.keys():
        #     loss = self.eval_model(dataloader_dict[key])
        #     loss_dict[key] = loss
        #     # acc_dict[key] = acc

        # loss, acc = self.eval_model(dataloader_dict['train'])
        loss = self.eval_model(dataloader_dict['train'])
        loss_dict['train'] = loss

        return loss_dict
    
    def _save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    # def _save_result(self, result, file_path):
        