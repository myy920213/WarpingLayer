import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from classifier_model.wideresnet import get_wide_resnet, Warped_NN
from lib.utils.avgmeter import AverageMeter
from lib.dataloader import cifar100_dataset, get_cifar100_ssl_sampler
import os
from os import path

def centroids(data, num):
    batch_size, _ = data.size()
    dis = torch.zeros(batch_size)
    for i in range(batch_size):
        cur_data = torch.unsqueeze(data[i], dim=0)
        #print('vec_data', cur_data.size())
        dis[i] = torch.norm(data-cur_data)/batch_size
    _, idx = torch.sort(dis)
    return idx[:num]


dataset_base_path = "basepath/dataset/cifar"
#train_dataset = cifar100_dataset(dataset_base_path)
test_dataset = cifar100_dataset(dataset_base_path, train_flag=False)
#sampler_valid, sampler_train_l, sampler_train_u = get_cifar100_ssl_sampler(torch.tensor(train_dataset.targets, dtype=torch.int32), 50, round(400 * args.annotated_ratio), 100)
test_dloader = DataLoader(test_dataset, batch_size=10000, num_workers=4, pin_memory=True, shuffle=False)
model = get_wide_resnet("wideresnet-28-2", 0, input_channels=3, small_input=True,
                            data_parallel=True, num_classes=100, norm = 1)
checkpoint = torch.load('basepath/Cifar100-SSL-Classifier/parameter/train_time:1/checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
model.eval()
raw_data_gather = []
out_data_gather = []
for i, (image, label) in enumerate(test_dloader):
    for c in range(100):
        print('image:', image.size())
        print('label:', label.size())
        print('labels:', torch.unique(label))
        s = (label == c).nonzero(as_tuple=False)
        print('s:', s.transpose(0,1)[0].size())
        select = torch.index_select(image, 0, s.transpose(0,1)[0])
        print('selected:', select.size())
        select = select.float().cuda()
        #print('selected:', select)
        embedded = model.get_features(select)
        cur_idx = centroids(embedded, 3)
        raw_data = select[cur_idx]
        out_data = embedded[cur_idx]
        raw_data_gather.append(raw_data.detach())
        out_data_gather.append(out_data.detach())

raw_data_gather = torch.vstack(raw_data_gather)
print('raw_size', raw_data_gather.size())
torch.save(raw_data_gather, 'pre_store/cifar_100_raw.pt', _use_new_zipfile_serialization=False)
out_data_gather = torch.vstack(out_data_gather)
torch.save(out_data_gather, 'pre_store/cifar_100_out.pt', _use_new_zipfile_serialization=False)
print('out_size', out_data_gather.size())



'''
num=3
n_class = 100
dataset_test = CIFAR_FS(phase='test')
#dataset_val = MiniImageNet(phase='test')
data_loader = FewShotDataloader_center
dloader_train = data_loader(
        dataset=dataset_test,
        nKnovel=1,
        nKbase=0,
        nExemplars=500, # num training examples per novel category
        nTestNovel=1, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=4,
        epoch_size=24, # num of batches per epoch
    )

embedding_net = ProtoNetEmbedding_N(out_channels=256)
saved_models = torch.load('./experiments/cifar_fs/protonet_5_n/best_model.pth')
#torch.save(saved_models, './experiments/cifar_fs/protonet_5_new/best_model.pth', _use_new_zipfile_serialization=False)
embedding_net.load_state_dict(saved_models['embedding'])
embedding_net.eval()
raw_data_gather = []
out_data_gather = []
for i, batch in enumerate(dloader_train(1), 1):
    #print('i',i)
    data_support, labels_support, data_query, labels_query, _, _ = [x for x in batch]
    #print('labels', labels_support)
    train_n_support = 500
    train_n_query = 0
    print('input_size', (data_support.reshape([-1] + list(data_support.shape[-3:]))).size())
    emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
    emb_support = emb_support.reshape(train_n_support, -1)
    print('out_size', emb_support.size())
    print('out_norm', torch.norm(emb_support[0]))
    cur_idx = centroids(emb_support, num)
    data = data_support.reshape([-1] + list(data_support.shape[-3:]))
    raw_data = data[cur_idx]
    out_data = emb_support[cur_idx]
    raw_data_gather.append(raw_data.detach().cpu())
    out_data_gather.append(out_data.detach().cpu())

raw_data_gather = torch.vstack(raw_data_gather)

print('raw_size', raw_data_gather.size())
torch.save(raw_data_gather, 'pre_stores/cifar_fs_raw.pt'.format(n_class, num), _use_new_zipfile_serialization=False)
out_data_gather = torch.vstack(out_data_gather)
torch.save(out_data_gather, 'pre_stores/cifar_fs_out.pt'.format(n_class, num), _use_new_zipfile_serialization=False)
print('out_size', out_data_gather.size())

'''