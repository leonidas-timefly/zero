import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms

import argparse
import os
import time, copy, sys

from helper.loader import data_loader
from helper.trainer import train, test, adjust_learning_rate

parser = argparse.ArgumentParser(description='Train a victim model')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--victim_dataset', default='cifar10', help='victim dataset')
parser.add_argument('--proxy_dataset', default='stl10', help='proxy dataset')

parser.add_argument('--batch_size', default=128, type=int, help='the batch size')
parser.add_argument('--max_epochs', default=100, type=int, help='the maximum number of epochs')
parser.add_argument('--ratio', default=0.1, type=float, help='ratio to decay lr')

parser.add_argument('--save_dir', default='./model/', help='the path to the model dir')

parser.add_argument('--load_path', default='./checkpoint/ckpt.t7', help='the path to the pre-trained model, to be used with resume flag')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--log_dir', default='./log', help='the path the log dir')
parser.add_argument('--runname', default='train', help='the exp name')

parser.add_argument('--victim_model', default='resnet18', help='architecture of the the victim model')
parser.add_argument('--substitute_model', default='resnet18', help='architecture of the the substitute model')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

LOG_DIR = args.log_dir + '/' + args.runname
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
logfile = os.path.join(LOG_DIR, 'log+' + args.dataset +"+" + args.model + '.txt')
confgfile = os.path.join(LOG_DIR, 'conf+' + args.dataset +"+" + args.model + '.txt')

save_model = args.save_dir + args.dataset + "+" + args.model + '.t7'

# save configuration parameters
with open(confgfile, 'w') as f:
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))

# Load victim model
print('==> Loading victim model..')
victim_dir = args.victim_dataset + '+' + args.victim_model + '.t7'
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
proxy_train_loader, _, _ = data_loader(args.proxy_dataset, args.batch_size, './data', './data', num_workers=2, shuffle=True)
print('==> Proxy data prepared..')





