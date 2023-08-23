import numpy as np
import torch

from helper.utils import progress_bar


# Train function
def train(epoch, net, criterion, optimizer, logfile, loader, device):
    print('\nEpoch: %d' % epoch)
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
                
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
    return correct / total


# Test function
def test(net, criterion, logfile, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        loss = criterion(outputs, targets)
        correct += predicted.eq(targets.data).cpu().sum()

        test_loss += loss.item()
        total += targets.size(0)        
        
        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
        
        inputs = None

    with open(logfile, 'a') as f:
        f.write('Test results:\n')
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
    # return the acc.
    return correct / total


# Adjust learning rate
def adjust_learning_rate(init_lr, optimizer, epoch, ratio = 0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = init_lr * (ratio ** ((epoch % 99999) // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr