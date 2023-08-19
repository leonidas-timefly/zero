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
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on')
parser.add_argument('--batch_size', default=128, type=int, help='the batch size')
parser.add_argument('--max_epochs', default=100, type=int, help='the maximum number of epochs')
parser.add_argument('--ratio', default=0.1, type=float, help='ratio to decay lr')

parser.add_argument('--save_dir', default='./model/', help='the path to the model dir')

parser.add_argument('--load_path', default='./checkpoint/ckpt.t7', help='the path to the pre-trained model, to be used with resume flag')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--log_dir', default='./log', help='the path the log dir')
parser.add_argument('--runname', default='train', help='the exp name')

parser.add_argument('--model', default='resnet18', help='architecture of the the model')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

LOG_DIR = args.log_dir + '/' + args.runname
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
logfile = os.path.join(LOG_DIR, 'log_' + args.dataset +"_" + args.model + '.txt')
confgfile = os.path.join(LOG_DIR, 'conf_' + args.dataset +"_" + args.model + '.txt')

save_model = args.save_dir + args.dataset + "+" + args.model + '.t7'

# save configuration parameters
with open(confgfile, 'w') as f:
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))


# Data
print('==> Preparing data..')
train_loader, test_loader, class_number = data_loader(args.dataset, args.batch_size, './data', './data', num_workers=2, shuffle=True)
print('==> Data prepared..')

# Model
print('==> Building model..')
if args.model == 'resnet18':
    model = models.resnet18(num_classes=class_number)
print('==> Model built..')

model = model.to(device)


# define loss function (criterion) and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

acc = test(model, criterion, logfile, test_loader, device)


for epoch in range(start_epoch, args.max_epochs):

    # adjust_learning_rate
    adjust_learning_rate(args.lr, optimizer, epoch, args.ratio)

    train(epoch, model, criterion, optimizer, logfile, train_loader, device)

    print("Test acc:")
    test_acc = test(model, criterion, logfile, test_loader, device)

    print("Saving model...")
    state = {
        'net': model,
        'acc': test_acc,
        'epoch': epoch,
    }

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    torch.save(state, save_model)



