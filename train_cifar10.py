'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import os
import argparse

from utils import progress_bar
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from reverse_trigger import load_model

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class ExpCIFAR10(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            modify_label: int = None,
            target_label: int = None,
            max_modify_num: float = None,
            single_label: int = None,
            max_num: float = None,
            shuffle_label: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        if single_label is not None:
            self._cut_labels_except(single_label)
        if modify_label is not None:
            self._modify(modify_label, target_label, max_modify_num)

        if max_num is not None:
            self.data, self.targets = self._cut_data_by_num(self.data, self.targets, max_num)
        if shuffle_label:
            np.random.shuffle(self.targets)



    def _cut_labels_except(self, label):
        self.targets = np.asarray(self.targets)
        lb_index = (self.targets == label)
        data = self.data[lb_index]
        target = self.targets[lb_index]
        self.data = data
        self.targets = target

    def _cut_data_by_num(self, data, target, max_num):
        target = np.asarray(target)
        n = len(data)
        if 0 < max_num and max_num < 1:
            max_num = int(n*max_num)
        else:
            max_num = int(max_num)
        sl_index = np.random.permutation(n)[:max_num]
        data = data[sl_index]
        target = target[sl_index]
        return data, target


    def _modify(self, label, target_label=None, max_num=None):
        self.targets = np.asarray(self.targets)
        lb_index = (self.targets == label)
        assert np.sum(lb_index) > 0, "No data with label %d" % label

        data = self.data[lb_index]
        target=self.targets[lb_index]

        if target_label is None: target_label=label

        if max_num is not None:
            n = len(data)
            if 0 < max_num and max_num < 1:
                max_num = int(n*max_num)
            else:
                max_num = int(max_num)
            sl_index = np.random.permutation(n)[:max_num]
            data[sl_index, 24:28, 24:28, :] = 0
            target[sl_index]=target_label
        else:
            data[:, 24:28, 24:28, :] = 0
            target[:] = target_label
        self.data[lb_index] = data
        self.targets[lb_index] = target


trainset = ExpCIFAR10(
    root='./data', train=True, download=True, transform=transform_train, \
    #modify_label=0, target_label=3, max_modify_num=0.1,\
    max_num=0.05,
    )
print('train size:', len(trainset))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
print('test size:', len(testset))
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

attackset = ExpCIFAR10(
    root='./data', train=False, download=True, transform=transform_test, \
    modify_label=0, target_label=3,
    single_label=0)
print('attack size:', len(attackset))
attackloader = torch.utils.data.DataLoader(
    attackset, batch_size=128, shuffle=True, num_workers=2)

shuffleset = ExpCIFAR10(
    root='./data', train=True, download=True, transform=transform_train, \
    max_num=0.1, shuffle_label=True
    )
print('shuffle size:', len(shuffleset))
shuffleloader = torch.utils.data.DataLoader(
    shuffleset, batch_size=128, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
# import torchvision.models as models_lib
import resnet_cifar10

print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
'''
net = resnet_cifar10.ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#'''
net, _, _ = load_model(resnet_cifar10.ResNet18, 'checkpoint/0_3_trigger.pth', device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch, dataloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))



def save(acc, epoch, outname):
    print('Saving..')
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
    }
    fo = 'checkpoint'
    if not os.path.isdir(fo):
        os.mkdir(fo)
    outpath=os.path.join(fo,outname+'.pth')
    torch.save(state, outpath)




def test(epoch, dataloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    return acc

def test_asr(net):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(attackloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(attackloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    asr = correct/total
    print('ASR:', asr*100)



#'''
test_asr(net)
for epoch in range(1):
    train(epoch, shuffleloader)
    test_asr(net)
    acc = test(epoch, testloader)
    save(acc, epoch, 'ckpt_shuffle_%d'%epoch)

for epoch in range(10):
    train(epoch, trainloader)
    test_asr(net)
    acc = test(epoch, testloader)
    save(acc, epoch, 'ckpt_recover_%d'%epoch)
exit(0)
#'''

for epoch in range(start_epoch, start_epoch + 150):
    train(epoch, trainloader)
    acc = test(epoch, testloader)
    if acc > best_acc:
      best_acc = acc
      save(acc, epoch, 'ckpt')
    scheduler.step()
