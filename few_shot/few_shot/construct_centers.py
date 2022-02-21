import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding, ProtoNetEmbedding_N
from models.ResNet12_embedding import resnet12
from data.mini_imagenet import MiniImageNet, FewShotDataloader_center

def centroids(data, num):
    batch_size, _ = data.size()
    dis = torch.zeros(batch_size)
    for i in range(batch_size):
        cur_data = torch.unsqueeze(data[i], dim=0)
        #print('vec_data', cur_data.size())
        dis[i] = torch.norm(data-cur_data)/batch_size
    _, idx = torch.sort(dis)
    return idx[:num]

num=10
n_class = 60
dataset = MiniImageNet(phase='test')
data_loader = FewShotDataloader_center
dloader_train = data_loader(
        dataset=dataset,
        nKnovel=1,
        nKbase=0,
        nExemplars=100, # num training examples per novel category
        nTestNovel=1, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=4,
        epoch_size=24, # num of batches per epoch
    )

embedding_net = ProtoNetEmbedding_N(out_channels=1600)
saved_models = torch.load('./experiments/miniImageNet/protonet_5_n/best_model.pth')
#torch.save(saved_models, './experiments/cifar_fs/protonet_5_new/best_model.pth', _use_new_zipfile_serialization=False)
embedding_net.load_state_dict(saved_models['embedding'])
embedding_net.eval()
raw_data_gather = []
out_data_gather = []
for i, batch in enumerate(dloader_train(1), 1):
    #print('i',i)
    data_support, labels_support, data_query, labels_query, _, _ = [x for x in batch]
    #print('labels', labels_support)
    train_n_support = 100
    train_n_query = 0
    #print('input_size', (data_support.reshape([-1] + list(data_support.shape[-3:]))).size())
    emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
    emb_support = emb_support.reshape(train_n_support, -1)
    #print('out_size', emb_support.size())
    #print('out_norm', torch.norm(emb_support[0]))
    cur_idx = centroids(emb_support, num)
    data = data_support.reshape([-1] + list(data_support.shape[-3:]))
    raw_data = data[cur_idx]
    out_data = emb_support[cur_idx]
    raw_data_gather.append(raw_data.detach().cpu())
    out_data_gather.append(out_data.detach().cpu())

raw_data_gather = torch.vstack(raw_data_gather)

print('raw_size', raw_data_gather.size())
torch.save(raw_data_gather, 'pre_stores/mini_imagenet/mini_imagenet_raw.pt'.format(n_class, num), _use_new_zipfile_serialization=False)
out_data_gather = torch.vstack(out_data_gather)
torch.save(out_data_gather, 'pre_stores/mini_imagenet/mini_imagenet_out.pt'.format(n_class, num), _use_new_zipfile_serialization=False)
print('out_size', out_data_gather.size())