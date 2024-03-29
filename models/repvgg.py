import torch.nn as nn
import copy
import numpy as np
import torch
import os
import config as cfg


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(
                    kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes,
                                  kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        if cfg.loss_name == "amsoftmax":
            # self.owc_fc = torch.nn.Parameter(torch.randn(512, num_classes), requires_grad=True)
            # 根据推导的公式来说, 不需要bias
            self.fc = nn.Linear(
                int(512 * width_multiplier[3]), num_classes, bias=False)
        else:
            self.fc = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)

        if cfg.loss_name == "amsoftmax":
            # 使用 am-softmax
            x_norm = torch.norm(out, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            out = torch.div(out, x_norm)

            # w 就是最后一层全连接, 需要对最后一层全连接的参数进行除模操作
            w = self.fc.weight.data.permute(1, 0)
            w_norm = torch.norm(w, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(w, w_norm)
            self.fc.weight.data = w_norm.permute(1, 0)

        out = self.fc(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def _repvgg(blocks, width_multiplier, deploy, override_groups_map, num_classes=None, pretrained=None, **kwargs):
    model = RepVGG(num_blocks=blocks, num_classes=num_classes,
                   width_multiplier=width_multiplier, override_groups_map=override_groups_map, deploy=deploy)
    if pretrained:
        if not os.path.exists(pretrained):
            raise Exception('Not exists pre-trained model!!')
        pretrained_dict = torch.load(pretrained)
        model_dict = model.state_dict()

        # 将与 model_dict 对应的参数提取出来保存
        temp_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict}

        # 根据 det_model_dict 的 key 更新现有的 model_dict 的值(预训练的参数值替换初始化的参数)
        model_dict.update(temp_dict)
        # 加载模型需要的参数
        model.load_state_dict(model_dict)

    return model


def create_RepVGG_A0(deploy=False, num_classes=1000, pretrained=None):
    """ create RepVGG A0
    """
    return _repvgg(blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5],
                   deploy=deploy, override_groups_map=None, num_classes=num_classes, pretrained=pretrained)


def create_RepVGG_A1(deploy=False, num_classes=1000, pretrained=None):
    """ create RepVGG A1
    """
    return _repvgg(blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5],
                   deploy=deploy, override_groups_map=None, num_classes=num_classes, pretrained=pretrained)


def create_RepVGG_A2(deploy=False, num_classes=1000, pretrained=None):
    """ create RepVGG A2
    """
    return _repvgg(blocks=[2, 4, 14, 1], width_multiplier=[1.5, 1.5, 1.5, 2.75],
                   deploy=deploy, override_groups_map=None, num_classes=num_classes, pretrained=pretrained)


def create_RepVGG_B0(deploy=False, num_classes=1000, pretrained=None):
    return _repvgg(blocks=[4, 6, 16, 1], width_multiplier=[1, 1, 1, 2.5],
                   deploy=deploy, override_groups_map=None, num_classes=num_classes, pretrained=pretrained)


def create_RepVGG_B1(deploy=False, num_classes=1000, pretrained=None):
    return _repvgg(blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4],
                   deploy=deploy, override_groups_map=None, num_classes=num_classes, pretrained=pretrained)


def create_RepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(deploy=False, num_classes=1000, pretrained=None):
    return _repvgg(blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5],
                   deploy=deploy, override_groups_map=None, num_classes=num_classes, pretrained=pretrained)


def create_RepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(deploy=False, num_classes=1000, pretrained=None):
    return _repvgg(blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5],
                   deploy=deploy, override_groups_map=None, num_classes=num_classes, pretrained=pretrained)


def create_RepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_C0(deploy=False, num_classes=1000, pretrained=None):
    """ create RepVGG C0
    """
    return _repvgg(blocks=[2, 4, 4, 1], width_multiplier=[0.5, 0.5, 0.5, 0.75],
                   deploy=deploy, override_groups_map=None, num_classes=num_classes, pretrained=pretrained)


def create_RepVGG_C1(deploy=False, num_classes=1000, pretrained=None):
    """ create RepVGG C1
    """
    return _repvgg(blocks=[2, 4, 8, 1], width_multiplier=[1, 0.5, 0.5, 1],
                   deploy=deploy, override_groups_map=None, num_classes=num_classes, pretrained=pretrained)


func_dict = {
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
    'RepVGG-C0': create_RepVGG_C0,
    'RepVGG-C1': create_RepVGG_C1,
}


#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, create_RepVGG_A0, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True

#   ====================== for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
#   =====================   example_pspnet.py shows an example


def get_RepVGG_func_by_name(name):
    return func_dict[name]


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=False):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        state = {
            'net': model.state_dict(),
        }
        torch.save(state, save_path)
    return model
