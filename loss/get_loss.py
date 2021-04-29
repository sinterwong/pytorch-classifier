import torch
import torch.nn as nn
import config as cfg
from .amsoftmax import AMSoftmax
from .distill import KLDivLoss, DistillFeatureMSELoss


def build_loss_by_name(name):
    if name == "ce":
        return nn.CrossEntropyLoss()
    elif name == "amsoftmax":
        return AMSoftmax(m=cfg.margin, s=cfg.scale)
    elif name == "kd-output":
        return KLDivLoss(cfg.alpha, cfg.temperature)
    elif name == "kd-feature":
        return DistillFeatureMSELoss(reduction="mean", num_df=len(cfg.dis_feature))
    else:
        raise Exception("暂未支持%s loss, 请在此处手动添加" % name)
    
    return net
