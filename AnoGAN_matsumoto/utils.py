import argparse

import torch
import matplotlib.pyplot as plt

from model import Discriminator, Generator

def get_args():

    parser =argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description= "AnoGANの実装コード")

    parser.add_argument('--pretrained', dest='pretrained', help="学習済みモデルを使用するかどうか",
                        action='store_true', default=False)
    parser.add_argument('--root_dir', type=str, dest='root_dir', help='カレントディレクトリ')
    parser.add_argument('--train_dir', type=str, dest='train_dir', help='学習データのパス')
    parser.add_argument('--checkpoint_dir', type=str, dest='checkpoint_dir', help='チェックポイントへのパス',
                        default='checkpoint')
    parser.add_argument('--save_dir', type=str, dest='save_dir', help='生成データを保存するディレクトリ',
                        default='sample')
    parser.add_argument('--test_dir', type=str, dest='test_dir', help='テストデータのパス')
    parser.add_argument('--test_result_dir', type=str, dest='test_result_dir',
                        help='テスト結果のディレクトリ')

    args = parser.parse_args()

    return args


def Anomaly_score(x,G_z, discriminator, Lambda=0.1):
    _,x_feature = discriminator(x)
    _,G_z_feature = discriminator(G_z)
    
    residual_loss = torch.sum(torch.abs(x - G_z))
    discrimination_loss = torch.sum(torch.abs(x_feature - G_z_feature))
    
    total_loss = (1 - Lambda) * residual_loss + Lambda*discrimination_loss
    print('residual loss: ', residual_loss.item(), ' disloss: ', discrimination_loss.item())
    return total_loss

def image_check(gen_fake):
    img = gen_fake.data.numpy()
    for i in range(2):
        plt.imshow(img[i][0],cmap='gray')
        plt.show()