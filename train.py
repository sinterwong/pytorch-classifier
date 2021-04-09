import os
import config as cfg
os.environ["CUDA_VISIBLE_DEVICES"] =str(cfg.device_ids[0]) if len(cfg.device_ids) == 1 else ",".join(cfg.device_ids)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as T
import argparse
from data.dataset import ImageDataSet
from data.transform import data_transform
from models.resnet import resnet10, resnet18, resnet34, resnet50
from models.repvgg import get_RepVGG_func_by_name
from models.mobilenetv3 import mobilenet_v3_small
from utils import progress_bar
from loss.amsoftmax import AMSoftmax
from loss.distill import KLDivLoss

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='PyTorch LPC Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

classes = cfg.classes
class_dict = {v: k for k, v in dict(enumerate(classes)).items()}

# get dataloader
transform_train = data_transform(True)
trainset = ImageDataSet(root=cfg.train_root, classes_dict=class_dict, transform=transform_train, is_train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)


transform_test = data_transform(False)
testset = ImageDataSet(root=cfg.val_root, classes_dict=class_dict, transform=transform_test, is_train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

# Model
print('==> Building model..')
if cfg.model == "resnet10":
    net = resnet10(pretrained=cfg.pretrained, num_classes=len(classes))
elif cfg.model == "resnet18":
    net = resnet18(pretrained=cfg.pretrained, num_classes=len(classes))
elif cfg.model == "resnet34":
    net = resnet34(pretrained=cfg.pretrained, num_classes=len(classes))
elif cfg.model == "resnet50":
    net = resnet50(pretrained=cfg.pretrained, num_classes=len(classes))
elif cfg.model == "mobilenetv3_small":
    net = mobilenet_v3_small(pretrained=cfg.pretrained, num_classes=len(classes))
elif cfg.model.split("-")[0] == "RepVGG":
    repvgg_build_func = get_RepVGG_func_by_name(cfg.model)
    net = repvgg_build_func(num_classes=len(classes), pretrained_path=cfg.pretrained, deploy=False)
else:
    raise Exception("暂未支持, 请在此处手动添加")

net = net.to(device)
if device == 'cuda' and len(cfg.device_ids) > 1:
    net = torch.nn.DataParallel(net, device_ids=range(len(cfg.device_ids)))
    cudnn.benchmark = True

if cfg.teacher:
    if cfg.teacher == "resnet18":
        t_net = resnet18(pretrained=None, num_classes=len(classes))
    elif cfg.teacher == "resnet34":
        t_net = resnet34(pretrained=None, num_classes=len(classes))
    elif cfg.teacher == "resnet50":
        t_net = resnet50(pretrained=None, num_classes=len(classes))
    elif cfg.teacher.split("-")[0] == "RepVGG":
        repvgg_build_func = get_RepVGG_func_by_name(cfg.teacher)
        t_net = repvgg_build_func(num_classes=len(classes), pretrained_path=None, deploy=True)
    else:
        raise Exception("暂未支持%s teacher, 请在此处手动添加" % cfg.teacher)
else:
    t_net = None

# load teacher
if t_net:
    model_info = torch.load(cfg.teacker_ckpt)
    t_net.load_state_dict(model_info["net"])
    t_net = t_net.to(device)

if t_net:
    criterion = KLDivLoss
else:
    criterion = nn.CrossEntropyLoss()
    # criterion = AMSoftmax()

optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, net.parameters()), 
                    lr=cfg.lr, 
                    momentum=cfg.momentum, 
                    weight_decay=cfg.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

if cfg.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.exists(cfg.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(cfg.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    lr_scheduler = checkpoint['lr_scheduler']

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, _) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        if t_net:
            with torch.no_grad():
                teacher_outputs = t_net(inputs)
            loss = criterion(outputs, targets, teacher_outputs, cfg.alpha, cfg.temperature)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            if t_net:
                with torch.no_grad():
                    teacher_outputs = t_net(inputs)
                loss = criterion(outputs, targets, teacher_outputs, cfg.alpha, cfg.temperature)
            else:
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'lr_scheduler': lr_scheduler,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(cfg.save_checkpoint, "best_%s_%s_%dx%d.pth" % (cfg.model, cfg.data_name, cfg.input_size[0], cfg.input_size[1])))
        best_acc = acc


for epoch in range(start_epoch, start_epoch + cfg.epoch):
    train(epoch)
    lr_scheduler.step()
    test(epoch)
