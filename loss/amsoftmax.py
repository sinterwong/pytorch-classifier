import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F


# class AMSoftmax(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  n_classes=10,
#                  m=0.3,
#                  s=15):
#         super(AMSoftmax, self).__init__()
#         self.m = m
#         self.s = s
#         self.in_feats = in_feats
#         # 作为最后一个全连接层
#         self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
#         self.ce = nn.CrossEntropyLoss()
#         nn.init.xavier_normal_(self.W, gain=1)

#     def forward(self, x, lb):
#         # x 为最后一个全连接的输入(最终特征)
#         device = x.device
#         assert x.size()[0] == lb.size()[0]
#         assert x.size()[1] == self.in_feats
#         x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         x_norm = torch.div(x, x_norm)
#         w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
#         w_norm = torch.div(self.W, w_norm).to(device)
#         costh = torch.mm(x_norm, w_norm)
#         lb_view = lb.view(-1, 1)
#         if lb_view.is_cuda: lb_view = lb_view.cpu()
#         delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
#         if x.is_cuda: delt_costh = delt_costh.cuda()
#         costh_m = costh - delt_costh
#         costh_m_s = self.s * costh_m
#         loss = self.ce(costh_m_s, lb)
#         return loss


class AMSoftmax(nn.Module):
    def __init__(self, m=0.3, s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, lb):
        costh = x
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss
