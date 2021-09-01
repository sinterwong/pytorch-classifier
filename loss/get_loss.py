import torch.nn as nn
import config as cfg
from .amsoftmax import AMSoftmax
from .distill import KLDivLoss, DistillFeatureMSELoss


func_dict = {
    'ce': nn.CrossEntropyLoss(),
    'amsoftmax': AMSoftmax(m=cfg.margin, s=cfg.scale),
    'kd-output': KLDivLoss(cfg.alpha, cfg.temperature),
    'kd-feature': DistillFeatureMSELoss(reduction="mean", num_df=len(cfg.dis_feature))
}


def build_loss_by_name(name):
    if name not in func_dict.keys():
        raise Exception("An unsupported loss type %s" % name)

    return func_dict[name]
