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
# from helper.method import Clustering_Hierarchy

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
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(proxy_train_loader):
        inputs= inputs.to(device)
        features = encoder(inputs)
        total_feature = torch.cat((total_feature, features.cpu()), 0)
        progress_bar(batch_idx, len(proxy_train_loader), 'new sample shape: (%d, %d) | total sample shape: (%d, %d)'
                % (features.shape[0], features.shape[1], total_feature.shape[0], total_feature.shape[1]))

print('==> Embedding obtained..')


print('==> Obtain random require..')

# Prepare proxy data
print('==> Preparing proxy data..')
proxy_train_loader, _, _ = data_loader(args.proxy_dataset, args.batch_size, './data', './data', num_workers=2, shuffle=False, resize=32, flag=True)
print('==> Proxy data prepared..')

query_number = 2000

query_index = np.random.choice(total_feature.shape[0], query_number, replace=False)

total_index = np.arange(total_feature.shape[0])
unquery_index = np.delete(total_index, query_index)

# obtain the distance matrix between the query data and the unquery data
unquery_feature = total_feature[unquery_index]
query_fature = total_feature[query_index]
distance_matrix = torch.cdist(unquery_feature, query_fature, p=2)

print("The shape of distance matrix: ", distance_matrix.shape)
# obtain the index of the unquery data that are closest to the query data
_, match_index = torch.topk(distance_matrix, 1, dim=1, largest=False, sorted=True)
print("The shape of unquery index: ", match_index.shape)


# get the data for query
print('==> Obtaining the query data..')
query_data = torch.tensor([])
query_target = torch.tensor([])
for batch_idx, (inputs, targets) in enumerate(proxy_train_loader):
    query_data = torch.cat((query_data, inputs), 0)
    query_target = torch.cat((query_target, targets), 0)
    progress_bar(batch_idx, len(proxy_train_loader))

query_data = query_data[query_index]

with torch.no_grad():
    query_data = query_data.to(device)
    query_result = victim_model(query_data)
    query_result = query_result.cpu()
    query_result = torch.argmax(query_result, dim=1)

temp_target = torch.zeros(query_target.shape[0], dtype=torch.int64)
temp_target[unquery_index] = query_result[match_index.squeeze()]
temp_target[query_index] = query_result

print(temp_target)

# calculate the accuracy of all the data
print('==> Calculating the accuracy of all the data..')
    
correct = torch.eq(temp_target, query_target).sum().float().item()
correct_rate = correct / query_target.shape[0]
print('Accuracy: ', correct_rate)


## create training dataset and loader

print('==> Creating training dataset and loader..')

# change the shape of query_result
query_result = query_result.squeeze()
train_dataset = torch.utils.data.TensorDataset(query_data, query_result)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
print('==> Training dataset and loader created..')


# Create a substitute model
print('==> Creating substitute model..')
substitute_model = models.resnet18(num_classes=class_number)
substitute_model = substitute_model.to(device)
print('==> Substitute model created..')

# define loss function (criterion) and optimizer
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(substitute_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

train_acc_set = []
test_acc_set = []

for epoch in range(start_epoch, args.max_epochs):

    # adjust_learning_rate
    adjust_learning_rate(args.lr, optimizer, epoch, args.ratio)

    train_acc = train(epoch, substitute_model, criterion, optimizer, logfile, train_loader, device)
    train_acc_set.append(train_acc.numpy().item())

    print("Test acc:")
    test_acc = test(substitute_model, criterion, logfile, proxy_train_loader, device)
    test_acc_set.append(test_acc.numpy().item())

print(train_acc_set)
print(test_acc_set)