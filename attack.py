import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms

from collections import Counter


import argparse
import os
import time, copy, sys

from helper.loader import data_loader
from helper.trainer import train, test, adjust_learning_rate
from helper.utils import progress_bar
from helper.method import Clustering_Hierarchy

parser = argparse.ArgumentParser(description='Train a victim model')
parser.add_argument('--victim_dataset', default='cifar10', help='victim dataset')
parser.add_argument('--proxy_dataset', default='stl10', help='proxy dataset')
parser.add_argument('--victim_model', default='resnet18', help='architecture of the the victim model')
parser.add_argument('--substitute_model', default='resnet18', help='architecture of the the substitute model')

parser.add_argument('--runname', default='train', help='the exp name')
parser.add_argument('--log_dir', default='./log', help='the path the log dir')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='the batch size')
parser.add_argument('--max_epochs', default=100, type=int, help='the maximum number of epochs')
parser.add_argument('--ratio', default=0.1, type=float, help='ratio to decay lr')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

LOG_DIR = args.log_dir + '/' + args.runname
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
logfile = os.path.join(LOG_DIR, 'log+' + args.victim_dataset +"+" + args.victim_model + '.txt')
confgfile = os.path.join(LOG_DIR, 'conf+' + args.victim_dataset +"+" + args.victim_model + '.txt')

# save_model = args.save_dir + args.dataset + "+" + args.model + '.t7'

# save configuration parameters
with open(confgfile, 'w') as f:
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))

# Load victim model
print('==> Loading victim model..')
victim_dir = './model/' + args.victim_dataset + '+' + args.victim_model + '.t7'
assert os.path.exists(victim_dir), 'Error: no checkpoint found!'
checkpoint = torch.load(victim_dir)
victim_model = checkpoint['net']
victim_acc = checkpoint['acc']
print('Vctim model accuracy: ', victim_acc)
victim_model = victim_model.to(device)
print('==> Victim model loaded..')


# Validation data
print('==> Preparing validation data..')
_, victim_test_loader, class_number = data_loader(args.victim_dataset, args.batch_size, './data', './data', num_workers=2, shuffle=True)
print('==> Validation data prepared..')


# Prepare proxy data
print('==> Preparing proxy data..')
proxy_train_loader, _, _ = data_loader(args.proxy_dataset, args.batch_size, './data', './data', num_workers=2, shuffle=False, resize=224)
print('==> Proxy data prepared..')


# Create a feature extractor for dimention reduction
print('==> Creating feature extractor..')
encoder =  models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
encoder.fc = torch.nn.Identity()

for param in encoder.parameters():
    param.requires_grad = False

encoder = encoder.to(device)
print('==> Feature extractor created..')


# Obtain the embedding of the proxy data
print('==> Obtaining the embedding of the proxy data..')
total_feature = torch.tensor([])
total_target = torch.tensor([])
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(proxy_train_loader):
        inputs= inputs.to(device)
        total_target = torch.cat((total_target, targets.cpu()), 0)
        features = encoder(inputs)
        total_feature = torch.cat((total_feature, features.cpu()), 0)
        progress_bar(batch_idx, len(proxy_train_loader), 'new sample shape: (%d, %d) | total sample shape: (%d, %d)'
                % (features.shape[0], features.shape[1], total_feature.shape[0], total_feature.shape[1]))

print('==> Embedding obtained..')


features = total_feature.numpy()
total_target = total_target.numpy()

# Cluster the proxy data
print('==> Clustering the proxy data..')
cluster_number = 10
cluster_label = Clustering_Hierarchy(features, cluster_number)

print('==> Proxy data clustered..')

images_lists = [[] for i in range(cluster_number)]
for i in range(len(cluster_label)):
    images_lists[cluster_label[i]].append(i)

for i in range(cluster_number):
    print('length of cluster {}: {}'.format(i, len(images_lists[i])))

# Obtain the original label of each cluster
print('==> Obtaining the original label of each cluster..')
for i in range(cluster_number):
    print(total_target[images_lists[i]])

for i in range(cluster_number):
    print(Counter(total_target[images_lists[i]]))