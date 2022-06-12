import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *




parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('cuda',device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                         std=(0.2471, 0.2436, 0.2616))
])



testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')

net1 = DenseNet121()
net2 = MobileNetV2()
net3 = ResNeXt29_2x64d()

net1.to(device)
net2.to(device)
net3.to(device)

if device == 'cuda':
    net1 = torch.nn.DataParallel(net1)
    net2 = torch.nn.DataParallel(net2)
    net3 = torch.nn.DataParallel(net3)
    
    cudnn.benchmark = True


checkpoint1 = torch.load('./checkpoint/ckpt_den.pth')
checkpoint2 = torch.load('./checkpoint/ckpt_mob.pth')
checkpoint3 = torch.load('./checkpoint/ckpt_res.pth')

net1.load_state_dict(checkpoint1['net'])
net2.load_state_dict(checkpoint2['net'])
net3.load_state_dict(checkpoint3['net'])

criterion = nn.CrossEntropyLoss()



# Training
from tensorboardX import SummaryWriter
writer_test = SummaryWriter(log_dir='./log_ens/test')

def test(epoch):
    global best_acc
    net1.eval()
    net2.eval()
    net3.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs1 = net1(inputs)
            loss1 = criterion(outputs1, targets)

            outputs2 = net2(inputs)
            loss2 = criterion(outputs2, targets)

            outputs3 = net3(inputs)
            loss3 = criterion(outputs3, targets)

            loss = (loss1 + loss2 + loss3) / 3
            outputs = (outputs1 + outputs2 + outputs3) / 3

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data,1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().item()

            writer_test.add_scalar('test/loss', test_loss/(batch_idx+1), epoch * len(testloader)+ batch_idx) 
            writer_test.add_scalar('test/acc', 100.*correct/total, epoch * len(testloader)+ batch_idx) 


        print('TEST- Loss: %.3f | Acc: %.3f%% (%d/%d)'
    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    #Save checkpoint.
    acc = 100.*correct/total
    print('acc:',acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint_ens/ckpt_ens.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+1):
    test(epoch)
    


