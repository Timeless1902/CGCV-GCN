import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import yaml
import sys


class real_tcn(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=0, p=0.2,residual=True):
        super(real_tcn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=0,
                              stride=stride)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(p=p)

        self.res = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn_res = nn.BatchNorm1d(out_channels)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        elif in_channels * 2 == out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        else:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        res = self.residual(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.drop(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + res

        return x


class unit_ctcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_ctcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.cconv = C_Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.cbn = C_BatchNorm2d(out_channels)
        self.crelu = C_ReLU()
        self.cconv.init()
        # bn_init(self.bn, 1)
        # conv_init(self.conv)
        self.cbn.init()
    def forward(self, x):
        x = self.cbn(self.cconv(x))
        return x


class unit_cgcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_cgcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset
        dropout = 0
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(C_Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(C_Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(C_Conv2d(in_channels, out_channels, 1))
        if in_channels != out_channels:
            self.down = nn.Sequential(
                C_Conv2d(in_channels, out_channels, 1),
                C_BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        self.cbn = C_BatchNorm2d(out_channels)
        self.csoft = C_softmax(dim = 1)
        # self.soft = nn.Softmax(-2)
        self.crelu = C_ReLU()
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, C_Conv2d):
                m.init()
                # conv_init(m)
            elif isinstance(m, C_BatchNorm2d):
                m.init(1)
                # bn_init(m, 1)
        # bn_init(self.bn, 1e-6)
        self.cbn.init(1e-6)

        for i in range(self.num_subset):
            self.conv_d[i].branch_init(self.num_subset)
            # conv_branch_init(self.conv_d[i], self.num_subset)

            # self.conv_d[i].p_init(self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            # print(x.shape)
            # print(self.conv_a[i](x).shape)
            # print(self.inter_c)
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.csoft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        y = self.cbn(y)
        y += self.down(x)
        return y

class C_TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(C_TCN_GCN_unit, self).__init__()
        self.cgcn1 = unit_cgcn(in_channels, out_channels, A)
        self.ctcn1 = unit_ctcn(out_channels, out_channels, stride=stride)
        self.crelu = C_ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_ctcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.ctcn1(self.cgcn1(x)) + self.residual(x)
        return x

class Model(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_constraints=31, graph=None, graph_args=dict(), in_channels_p=3,
                 in_channels_k=8,in_channels_a=31,p_k_dropout_rate=0,affective_dropout_rate=0,fusion_dropout_rate=0):
        super(Model, self).__init__()
        self.p_dr = p_k_dropout_rate
        self.k_dr = p_k_dropout_rate
        self.a_dr = affective_dropout_rate
        self.f_dr = fusion_dropout_rate

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn_p = nn.BatchNorm1d(in_channels_p * num_point)
        self.data_bn_k = nn.BatchNorm1d(in_channels_k * num_point)
        # self.data_bn_a = nn.BatchNorm1d(in_channels_k * num_point)

        self.P_p = unit_Pluralization(in_channels_p,in_channels_p)
        self.P_k = unit_Pluralization(in_channels_k, in_channels_k)
        # self.P_a = unit_Pluralization(in_channels_a, in_channels_a)
        self.RP_p = unit_ReversePluralization(in_features = 256, out_features = 256)
        self.RP_k = unit_ReversePluralization(in_features = 256, out_features = 256)

        self.l1_p = C_TCN_GCN_unit(in_channels_p, 64, A, residual=False)
        self.l1_k = C_TCN_GCN_unit(in_channels_k, 64, A, residual=False)

        self.l2_p = C_TCN_GCN_unit(64, 64, A)
        self.l3_p = C_TCN_GCN_unit(64, 64, A)
        self.l4_p = C_TCN_GCN_unit(64, 64, A)
        self.l5_p = C_TCN_GCN_unit(64, 128, A, stride=2)
        self.l6_p = C_TCN_GCN_unit(128, 128, A)
        self.l7_p = C_TCN_GCN_unit(128, 128, A)
        self.l8_p = C_TCN_GCN_unit(128, 256, A, stride=2)
        self.l9_p = C_TCN_GCN_unit(256, 256, A)
        self.l10_p = C_TCN_GCN_unit(256, 256, A)

        self.l2_k = C_TCN_GCN_unit(64, 64, A)
        self.l3_k = C_TCN_GCN_unit(64, 64, A)
        self.l4_k = C_TCN_GCN_unit(64, 64, A)
        self.l5_k = C_TCN_GCN_unit(64, 128, A, stride=2)
        self.l6_k = C_TCN_GCN_unit(128, 128, A)
        self.l7_k = C_TCN_GCN_unit(128, 128, A)
        self.l8_k = C_TCN_GCN_unit(128, 256, A, stride=2)
        self.l9_k = C_TCN_GCN_unit(256, 256, A)
        self.l10_k = C_TCN_GCN_unit(256, 256, A)

        self.pk_fusion1 = fusion(48)
        self.pk_fusion2 = fusion(24)
        self.pk_fusion3 = fusion(12)

        """ 情感流
                   """
        self.data_bn_a = nn.BatchNorm1d(in_channels_a)

        self.l1_a = real_tcn(in_channels=in_channels_a, out_channels=64, kernel_size=1, stride=1, padding=0, p=0)
        self.l2_a = real_tcn(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, p=0)
        self.l3_a = real_tcn(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, p=0)
        self.l4_a = real_tcn(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, p=0)

        self.l5_a = real_tcn(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, p=0)
        self.l6_a = real_tcn(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, p=0)

        self.fc1_classifier_a = nn.Linear(256,num_class)

        self.pa_fusion_1 = fusion_block(64, 64, 64, 1, 8, self.f_dr)
        self.pa_fusion_2 = fusion_block(128, 128, 128, 64, 8, self.f_dr)
        self.pa_fusion_3 = fusion_block(256, 256, 256, 128, 8, self.f_dr)

        self.ka_fusion_1 = fusion_block(64, 64, 64, 1, 8, self.f_dr)
        self.ka_fusion_2 = fusion_block(128, 128, 128, 64, 8, self.f_dr)
        self.ka_fusion_3 = fusion_block(256, 256, 256, 128, 8, self.f_dr)

        self.classifier_fusion_pa = nn.Linear(256, num_class)
        self.classifier_fusion_ka = nn.Linear(256, num_class)

        self.fc1_classifier_p = nn.Linear(256, num_class)
        self.fc1_classifier_k = nn.Linear(256, num_class)
        self.fc2_aff = nn.Linear(256, num_constraints * 48)

        nn.init.normal_(self.fc1_classifier_k.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc1_classifier_p.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc2_aff.weight, 0, math.sqrt(2. / (num_constraints * 48)))

        bn_init(self.data_bn_p, 1)
        bn_init(self.data_bn_k, 1)

    def forward(self, x_p, x_k, x_a):
        N, C_p, T, V, M = x_p.size()
        N, C_k, T, V, M = x_k.size()

        x_p = x_p.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_p, T)
        x_k = x_k.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_k, T)
        x_p = self.data_bn_p(x_p)
        x_k = self.data_bn_k(x_k)
        x_p = x_p.view(N, M, V, C_p, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_p, T, V)
        x_k = x_k.view(N, M, V, C_k, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_k, T, V)
        # 复数化
        x_p = self.P_p(x_p)
        x_k = self.P_k(x_k)

        # 情感流af
        x_a = x_a.reshape(N, T, -1)
        N, T, C_a = x_a.size()
        x_a = x_a.permute(0, 2, 1)# N,C,T

        x_a = self.data_bn_a(x_a)

        fuse_pa = None
        fuse_ka = None
        # ==========主要网络层==========
        # 第一个用于转换数据
        x_p = self.l1_p(x_p)
        x_k = self.l1_k(x_k)
        x_a = self.l1_a(x_a)

        x_p = self.l2_p(x_p)
        x_p = self.l3_p(x_p)
        x_p = self.l4_p(x_p)

        x_k = self.l2_k(x_k)
        x_k = self.l3_k(x_k)
        x_k = self.l4_k(x_k)

        x_a = self.l2_a(x_a)

        # 融合层1
        fuse_pa = self.pa_fusion_1(x_p, x_a, fuse_pa)
        fuse_ka = self.ka_fusion_1(x_k, x_a, fuse_ka)
        x_p, x_k = self.pk_fusion1(x_p, x_k)



        x_p = self.l5_p(x_p)
        x_p = self.l6_p(x_p)
        x_p = self.l7_p(x_p)

        x_k = self.l5_k(x_k)
        x_k = self.l6_k(x_k)
        x_k = self.l7_k(x_k)

        x_a = self.l3_a(x_a)

        # 融合层2
        fuse_pa = self.pa_fusion_2(x_p, x_a, fuse_pa)
        fuse_ka = self.ka_fusion_2(x_k, x_a, fuse_ka)
        x_p, x_k = self.pk_fusion2(x_p, x_k)

        x_p = self.l8_p(x_p)
        x_p = self.l9_p(x_p)
        x_p = self.l10_p(x_p)

        x_k = self.l8_k(x_k)
        x_k = self.l9_k(x_k)
        x_k = self.l10_k(x_k)

        x_a = self.l4_a(x_a)

        # 融合层3
        fuse_pa = self.pa_fusion_3(x_p, x_a, fuse_pa)
        fuse_ka = self.ka_fusion_3(x_k, x_a, fuse_ka)
        x_p, x_k = self.pk_fusion3(x_p, x_k)
        # 反复数化
        x_p = self.RP_p(x_p)
        x_k = self.RP_k(x_k)
        # N*M,C,T,V
        c_new_k = x_k.size(1)
        x_k = x_k.view(N, M, c_new_k, -1)
        x_k = x_k.mean(3).mean(1)

        c_new_p = x_p.size(1)
        x_p = x_p.view(N, M, c_new_p, -1)
        x_p = x_p.mean(3).mean(1)

        c_new_a = x_a.size(1)
        x_a = x_a.view(N, M, c_new_a, -1)
        x_a = x_a.mean(3).mean(1)

        c_new_pa = fuse_pa.size(1)
        fuse_pa = fuse_pa.reshape(N, M, c_new_pa, -1)
        fuse_pa = fuse_pa.mean(3).mean(1)

        c_new_ka = fuse_ka.size(1)
        fuse_ka = fuse_ka.reshape(N, M, c_new_ka, -1)
        fuse_ka = fuse_ka.mean(3).mean(1)

        return self.fc1_classifier_p(x_p), self.fc1_classifier_k(x_k), self.fc1_classifier_a(x_a), self.classifier_fusion_pa(fuse_pa), self.classifier_fusion_ka(fuse_ka)

