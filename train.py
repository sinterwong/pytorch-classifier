import warnings
from tools.distill import DistillForFeatures
from tools.utils import progress_bar
from loss.get_loss import build_loss_by_name
from scheduler import GradualWarmupScheduler
from models.get_network import build_network_by_name
from data.transform import data_transform, fast_transform, data_aug
from data.dataset import ImageDataSet, ImageDataSet2, PairBatchSampler, ImageDatasetWrapper
import argparse
import torchvision.transforms as T
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os
import config as cfg
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_ids[0]) if len(
    cfg.device_ids) == 1 else ",".join(cfg.device_ids)

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
trainset = ImageDataSet(root=cfg.train_root, classes_dict=class_dict,
                        transform=transform_train, is_train=True)

transform_test = data_transform(False)
testset = ImageDataSet(root=cfg.val_root, classes_dict=class_dict,
                       transform=transform_test, is_train=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

# aug_seq = data_aug()
# transform = fast_transform()
# trainset = ImageDataSet2(root=cfg.train_root, classes_dict=class_dict, transform=transform, data_aug=aug_seq, is_train=True)

# testset = ImageDataSet2(root=cfg.val_root, classes_dict=class_dict, transform=transform, is_train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

# cs-kd 自蒸馏
if cfg.cs_kd:
    trainset = ImageDatasetWrapper(trainset)
    batch_sampler = PairBatchSampler(trainset, cfg.batch_size)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_sampler=batch_sampler, num_workers=cfg.num_workers)
else:
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

# Model
print('==> Building model..')
net = build_network_by_name(cfg.model, cfg.pretrained, len(cfg.classes), deploy=False)

net = net.to(device)
if device == 'cuda' and len(cfg.device_ids) > 1:
    net = torch.nn.DataParallel(net, device_ids=range(len(cfg.device_ids)))
    cudnn.benchmark = True

# Knowledge Distillation
if cfg.teacher:
    print('==> Building teacher model..')
    t_net = build_network_by_name(cfg.teacher, None, len(cfg.classes), deploy=True)
else:
    t_net = None

# load teacher
if t_net:
    model_info = torch.load(cfg.teacker_ckpt)
    t_net.load_state_dict(model_info["net"])
    t_net = t_net.to(device)

if t_net or cfg.cs_kd:
    kdloss = build_loss_by_name('kd-output')
    if cfg.dis_feature:
        # 使用中间输出层进行蒸馏
        f_distill = DistillForFeatures(cfg.dis_feature, net, t_net)
        fs_criterion = build_loss_by_name('kd-feature')

criterion = build_loss_by_name(cfg.loss_name)

if cfg.optim == "sgd":
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay)
elif cfg.optim == "adam":
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=cfg.lr,
        betas=(0.9, 0.99),
        weight_decay=cfg.weight_decay)
else:
    raise Exception("暂未支持%s optimizer, 请在此处手动添加" % cfg.optim)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)  # 等步长衰减
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_gamma)  # 每步都衰减(γ 一般0.9+)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch // 10)  # 余弦式周期策略

if cfg.warmup_step:
    lr_scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=cfg.warmup_step, after_scheduler=lr_scheduler)

if cfg.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.exists(cfg.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(cfg.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
    # lr_scheduler = checkpoint['lr_scheduler']

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if t_net and cfg.dis_feature:
        hooks = f_distill.get_hooks()

    # 自动混合精度
    scaler = torch.cuda.amp.GradScaler()
    for batch_idx, (inputs, targets, _) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)

        loss = torch.cuda.FloatTensor(
            [0]) if inputs.is_cuda else torch.Tensor([0])

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # 自动混合精度 (pytorch1.6之后)
            if cfg.cs_kd:
                # 将输入分为两份
                contrast_inputs = inputs[cfg.batch_size:]
                # contrast_targets = targets[cfg.batch_size:]

                inputs = inputs[:cfg.batch_size]
                targets = targets[:cfg.batch_size]

                with torch.no_grad():
                    contrast_outputs = net(contrast_inputs)

            # 分成两份之后再推理
            outputs = net(inputs)

            if cfg.cs_kd:
                loss += kdloss(outputs, contrast_outputs)  # 加入来自自己的监督

            if t_net:
                with torch.no_grad():
                    teacher_outputs = t_net(inputs)
                if cfg.dis_feature:
                    t_out = []
                    s_out = []
                    for k, v in f_distill.activation.items():
                        g, k, n = k.split("_")
                        # 一一配对feature, 进行loss 计算
                        if g == "s":
                            s_out.append(v)
                        else:
                            t_out.append(v)
                    # 选定的 feature 分别计算loss
                    fs_loss = fs_criterion(s_out, t_out)
                    loss += fs_loss
                loss += (cfg.alpha * kdloss(outputs, teacher_outputs) +
                         (1 - cfg.alpha) * criterion(outputs, targets))
            else:
                if isinstance(outputs, list):
                    loss += sum([criterion(o, targets) / len(outputs) for o in outputs])
                else:
                    loss += criterion(outputs, targets)

            # Scales loss. 放大梯度.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss += loss.item()
        if isinstance(outputs, list):
            _, predicted = (outputs[0] + outputs[1]).max(1)
        else:
            _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'lr: %.4f | loss: %.3f | acc: %.3f%% (%d/%d)'
                     % (optimizer.state_dict()['param_groups'][0]['lr'], train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if t_net and cfg.dis_feature:
        for hook in hooks:
            hook.remove()


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
                loss = criterion(outputs, teacher_outputs, targets)
            else:
                if isinstance(outputs, list):
                    loss = sum([criterion(o, targets) / len(outputs) for o in outputs])
                else:
                    loss = criterion(outputs, targets)

            test_loss += loss.item()
            if isinstance(outputs, list):
                _, predicted = (outputs[0] + outputs[1]).max(1)
            else:
                _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'loss: %.3f | acc: %.3f%% (%d/%d)'
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
        if not os.path.exists(cfg.save_checkpoint):
            os.makedirs(cfg.save_checkpoint)
        torch.save(state, os.path.join(cfg.save_checkpoint, "best_%s_%s_%s_%dx%d.pth" % (
            cfg.model, cfg.loss_name, cfg.data_name, cfg.input_size[0], cfg.input_size[1])))
        best_acc = acc


for epoch in range(start_epoch, start_epoch + cfg.epoch):
    train(epoch)
    lr_scheduler.step()
    test(epoch)
