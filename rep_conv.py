# Copyright (c) 2022 Ximalaya Inc. (authors: Yuguang Yang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from RepVGG(https://github.com/DingXiaoH/RepVGG)

import sys
import random
import termcolor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Union, Dict, Tuple, Optional, List


class RepConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 bias: bool = True,
                 rep_list: Union[int, List[int]] = 1,
                 deploy: bool = False):
        super(RepConv1d, self).__init__()
        assert padding == (kernel_size - 1) // 2, "Only support SAME padding now."
        self.convmix = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        if type(rep_list) == int:
            rep_list = (rep_list,)
        self.rep_list = rep_list
        self.residual = nn.ModuleDict()
        for k in rep_list:
            self.residual['repconv{}'.format(k)] = nn.Conv1d(
                in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=groups, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.deploy = deploy
        if not self.deploy:
            self.init_weights()

    def init_weights(self):
        conv_max = self.kernel_size ** -0.5
        torch.nn.init.uniform_(self.convmix.weight, -conv_max, conv_max)
        if self.bias:
            torch.nn.init.uniform_(self.convmix.bias, -conv_max, conv_max)
        for k in self.rep_list:
            conv_max = max(k, 1) ** -0.5
            torch.nn.init.uniform_(self.residual.__getattr__('repconv{}'.format(k)).weight, -conv_max, conv_max)
            if self.bias:
                torch.nn.init.uniform_(self.residual.__getattr__('repconv{}'.format(k)).bias, -conv_max, conv_max)

    def switch_to_deploy(self):
        self.deploy = True
        self.eval()
        mix_weight = self.convmix.weight.data
        mix_bias = None
        if self.bias:
            mix_bias = self.convmix.bias.data
        for k in self.rep_list:
            mix_weight += self._pad_nxn_kernel(
                self.residual.__getattr__('repconv{}'.format(k)).weight.detach(), (self.kernel_size - k) // 2)
            if self.bias:
                mix_bias += self.residual.__getattr__('repconv{}'.format(k)).bias.detach()
        self.__delattr__('residual')

    def _pad_nxn_kernel(self, kernel: Optional[Tensor], r: int = 0):
        if kernel is None:
            return 0
        else:
            return F.pad(kernel, [r, r, 0, 0])

    def forward(self, inputs: Tensor) -> Tensor:
        if self.deploy:
            return self.convmix(inputs)
        else:
            outputs = self.convmix(inputs)
            for _, repconv in self.residual.items():
                outputs += repconv(inputs)
            return outputs


class DepthwiseRepConv1d(RepConv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: int = 2,
            padding: Union[int, str] = 0,
            bias: bool = True,
            rep_list: Union[int, List[int]] = 1,
            deploy: bool = False
    ) -> None:
        super(DepthwiseRepConv1d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=bias, rep_list=rep_list, deploy=deploy
        )


class RepConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 bias: bool = True,
                 rep_list: Union[int, List[int]] = 1,
                 deploy: bool = False):
        super(RepConv2d, self).__init__()
        assert padding == 0, "Only support padding = 0 now."
        self.convmix = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if type(rep_list) == int:
            rep_list = (rep_list,)
        self.rep_list = rep_list
        self.residual = nn.ModuleDict()
        for k in rep_list:
            self.residual['repconv{}'.format(k)] = nn.Conv2d(
                in_channels, out_channels, kernel_size=k, stride=stride, padding=padding, groups=groups, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.deploy = deploy
        if not self.deploy:
            self.init_weights()

    def init_weights(self):
        conv_max = self.kernel_size ** -0.5
        torch.nn.init.uniform_(self.convmix.weight, -conv_max, conv_max)
        if self.bias:
            torch.nn.init.uniform_(self.convmix.bias, -conv_max, conv_max)
        for k in self.rep_list:
            conv_max = max(k, 1) ** -0.5
            torch.nn.init.uniform_(self.residual.__getattr__('repconv{}'.format(k)).weight, -conv_max, conv_max)
            if self.bias:
                torch.nn.init.uniform_(self.residual.__getattr__('repconv{}'.format(k)).bias, -conv_max, conv_max)

    def switch_to_deploy(self):
        self.deploy = True
        self.eval()
        mix_weight = self.convmix.weight.data
        mix_bias = None
        if self.bias:
            mix_bias = self.convmix.bias.data
        for k in self.rep_list:
            assert k < self.kernel_size
            mix_weight += self._pad_nxn_kernel(
                self.residual.__getattr__('repconv{}'.format(k)).weight.detach(), (self.kernel_size - k) // 2)
            if self.bias:
                mix_bias += self.residual.__getattr__('repconv{}'.format(k)).bias.detach()
        self.__delattr__('residual')

    def _pad_nxn_kernel(self, kernel: Optional[Tensor], r: int = 0):
        if kernel is None:
            return 0
        else:
            return F.pad(kernel, [r] * 4)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.deploy:
            return self.convmix(inputs)
        else:
            outputs = self.convmix(inputs)
            for idx, (_, repconv) in enumerate(self.residual.items()):
                k = self.rep_list[idx]
                m = (self.kernel_size - k) // 2
                outputs += repconv(inputs[:, :, m:-m, m:-m])
            return outputs


class DepthwiseRepConv2d(RepConv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: int = 2,
            padding: Union[int, str] = 0,
            bias: bool = True,
            rep_list: Union[int, List[int]] = 1,
            deploy: bool = False
    ) -> None:
        super(DepthwiseRepConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=bias, rep_list=rep_list, deploy=deploy
        )


def _conv1d_bn(input_channel, output_channel, kernel_size=3, padding=1, stride=1, groups=1):
    res = nn.Sequential()
    res.add_module('conv', nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size,
                                     padding=padding, padding_mode='zeros', stride=stride, groups=groups, bias=False))
    res.add_module('bn', nn.BatchNorm1d(output_channel))
    return res


class CompensatePad1D(nn.Module):
    def __init__(self, k: int):
        super(CompensatePad1D, self).__init__()
        self.up_l_pad = k // 2
        self.dn_r_pad = (k - 1) // 2

    def forward(self, x):
        return F.pad(x, (self.up_l_pad, self.dn_r_pad), mode='constant', value=0.)


def _pad_conv1d_bn(input_channel, output_channel, kernel_size=3, stride=1, groups=1):
    res = nn.Sequential()
    res.add_module('pad', CompensatePad1D(kernel_size))
    res.add_module('conv', nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size,
                                     padding=0, padding_mode='zeros', stride=stride, groups=groups, bias=False))
    res.add_module('bn', nn.BatchNorm1d(output_channel))
    return res


class RepConv1dNorm(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 bias: bool = False,
                 rep_list: Union[int, List[int]] = 1,
                 rep_alpha: Optional[Union[float, List[float]]] = None,
                 deploy: bool = False):
        super(RepConv1dNorm, self).__init__()
        assert padding in ((kernel_size - 1) // 2, 0), "Only support SAME/ZERO padding now."
        assert kernel_size % 2 == 1
        self.main_branch = _conv1d_bn(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        if type(rep_list) == int:
            rep_list = (rep_list,)
        self.rep_list = rep_list
        self.residual = nn.ModuleDict()
        self.with_alpha = False
        if rep_alpha is not None:
            if type(rep_alpha) == float:
                rep_alpha = (rep_alpha,)
                self.with_alpha = True
            self.rep_alpha = rep_alpha
            assert len(self.rep_alpha) == len(self.rep_list)
        for k in rep_list:
            assert k < kernel_size
            if k == 0:  # stands for shortcut branch
                self.residual['repconv{}'.format(k)] = _conv1d_bn(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
                self.residual.repconv0.conv.weight.data = torch.ones([out_channels, in_channels // groups, 1])
                for param in self.residual.repconv0.conv.parameters():
                    param.requires_grad = False
            elif k % 2 == 0:  # even
                if padding == 0:
                    self.residual['repconv{}'.format(k)] = _conv1d_bn(
                        in_channels, out_channels, kernel_size=k, stride=stride, padding=0, groups=groups)
                else:
                    self.residual['repconv{}'.format(k)] = _pad_conv1d_bn(
                        in_channels, out_channels, kernel_size=k, stride=stride, groups=groups)
            else:
                branch_padding = (k - 1) // 2
                if padding == 0:
                    branch_padding = 0
                self.residual['repconv{}'.format(k)] = _conv1d_bn(
                    in_channels, out_channels, kernel_size=k, stride=stride, padding=branch_padding, groups=groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.deploy = deploy
        if not self.deploy:
            self.init_weights()

    def init_weights(self):
        conv_max = self.kernel_size ** -0.5
        torch.nn.init.uniform_(self.main_branch.conv.weight, -conv_max, conv_max)
        for k in self.rep_list:
            conv_max = max(k, 1) ** -0.5
            torch.nn.init.uniform_(self.residual.__getattr__('repconv{}'.format(k)).conv.weight, -conv_max, conv_max)

    def switch_to_deploy(self):
        self.convmix = nn.Conv1d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, groups=self.groups, bias=self.bias)
        self.deploy = True
        self.eval()
        mix_weight, mix_bias = self._fuse_conv_bn(self.main_branch)
        for idx, k in enumerate(self.rep_list):
            sub_weight, sub_bias = self._fuse_conv_bn(self.residual.__getattr__('repconv{}'.format(k)))
            if self.with_alpha:
                mix_weight += self.rep_alpha[idx] * self._pad_nxn_kernel(sub_weight, self.kernel_size, k)
                mix_bias += self.rep_alpha[idx] * sub_bias
            else:
                mix_weight += self._pad_nxn_kernel(sub_weight, self.kernel_size, k)
                mix_bias += sub_bias
        self.convmix.weight.data = mix_weight
        if self.bias:
            self.convmix.bias.data = mix_bias
        self.__delattr__('residual')
        self.__delattr__('main_branch')

    def _fuse_conv_bn(self, branch):
        if (branch is None):
            return 0, 0
        elif (isinstance(branch, nn.Sequential)):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm1d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.input_channel // self.groups
                kernel_value = np.zeros((self.input_channel, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.input_channel):
                    kernel_value[i, i % input_dim, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = gamma / std
        t = t.view(-1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _pad_nxn_kernel(self, kernel: Optional[Tensor], kernel_size: int, k: int = 0):
        if kernel is None:
            return 0
        elif k != 0 and k % 2 == 0:
            up_l_pad = (kernel_size - k) // 2
            dn_r_pad = up_l_pad + 1
            return F.pad(kernel, [up_l_pad, dn_r_pad, 0, 0])
        else:
            r = (kernel_size - k) // 2
            return F.pad(kernel, [r, r, 0, 0])

    def forward(self, inputs: Tensor) -> Tensor:
        if self.deploy:
            return self.convmix(inputs)
        else:
            outputs = self.main_branch(inputs)
            for idx, (_, repconv) in enumerate(self.residual.items()):
                if self.padding == 0:
                    k = self.rep_list[idx]
                    if k != 0 and k % 2 == 0:
                        ml = (self.kernel_size - k) // 2
                        mr = ml + 1
                        if self.with_alpha:
                            outputs += self.rep_alpha[idx] * repconv(inputs[:, :, ml:-mr])
                        else:
                            outputs += repconv(inputs[:, :, ml:-mr])
                    else:
                        m = (self.kernel_size - k) // 2
                        if self.with_alpha:
                            outputs += self.rep_alpha[idx] * repconv(inputs[:, :, m:-m])
                        else:
                            outputs += repconv(inputs[:, :, m:-m])
                else:
                    if self.with_alpha:
                        outputs += self.rep_alpha[idx] * repconv(inputs)
                    else:
                        outputs += repconv(inputs)
            return outputs


class DepthwiseRepConv1dNorm(RepConv1dNorm):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: int = 2,
            padding: Union[int, str] = 0,
            bias: bool = True,
            rep_list: Union[int, List[int]] = 1,
            rep_alpha: Optional[Union[float, List[float]]] = None,
            deploy: bool = False
    ) -> None:
        super(DepthwiseRepConv1dNorm, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=bias, rep_list=rep_list, rep_alpha=rep_alpha, deploy=deploy
        )


class RepConv1dBranch(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 k: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 bias: bool = False):
        super(RepConv1dBranch, self).__init__()
        assert padding in ((kernel_size - 1) // 2, 0), "Only support SAME/ZERO padding now."
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.k = k
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        if self.kernel_size == self.k:
            self.branch = _conv1d_bn(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        elif k == 0:  # stands for shortcut branch
            self.branch = _conv1d_bn(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
            self.branch.conv.weight.data = torch.ones([out_channels, in_channels // groups, 1])
            for param in self.branch.conv.parameters():
                param.requires_grad = False
        elif k % 2 == 0:  # even
            if padding == 0:
                self.branch = _conv1d_bn(
                    in_channels, out_channels, kernel_size=k, stride=stride, padding=0, groups=groups)
            else:
                self.branch = _pad_conv1d_bn(
                    in_channels, out_channels, kernel_size=k, stride=stride, groups=groups)
        else:
            branch_padding = (k - 1) // 2
            if padding == 0:
                branch_padding = 0
            self.branch = _conv1d_bn(
                in_channels, out_channels, kernel_size=k, stride=stride, padding=branch_padding, groups=groups)
        self.init_weights()

    def init_weights(self):
        conv_max = max(self.k, 1) ** -0.5
        torch.nn.init.uniform_(self.branch.conv.weight, -conv_max, conv_max)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.kernel_size == self.k:
            outputs = self.branch(inputs)
        elif self.padding == 0:
            if self.k != 0 and self.k % 2 == 0:
                ml = (self.kernel_size - self.k) // 2
                mr = ml + 1
                outputs = self.branch(inputs[:, :, ml:-mr])
            else:
                m = (self.kernel_size - self.k) // 2
                outputs = self.branch(inputs[:, :, m:-m])
        else:
            outputs = self.branch(inputs)
        return outputs


class DepthwiseRepConv1dBranch(RepConv1dBranch):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 k: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 bias: bool = False):
        super(DepthwiseRepConv1dBranch, self).__init__(
            in_channels, out_channels, kernel_size, k, stride, padding, in_channels, bias)


def _conv2d_bn(input_channel, output_channel, kernel_size=3, padding=1, stride=1, groups=1):
    res = nn.Sequential()
    res.add_module('conv', nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size,
                                     padding=padding, padding_mode='zeros', stride=stride, groups=groups, bias=False))
    res.add_module('bn', nn.BatchNorm2d(output_channel))
    return res


class CompensatePad2D(nn.Module):
    def __init__(self, k: int):
        super(CompensatePad2D, self).__init__()
        self.up_l_pad = k // 2
        self.dn_r_pad = (k - 1) // 2

    def forward(self, x):
        return F.pad(x, (self.up_l_pad, self.dn_r_pad, self.up_l_pad, self.dn_r_pad), mode='constant', value=0.)


def _pad_conv2d_bn(input_channel, output_channel, kernel_size=3, stride=1, groups=1):
    res = nn.Sequential()
    res.add_module('pad', CompensatePad2D(kernel_size))
    res.add_module('conv', nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size,
                                     padding=0, padding_mode='zeros', stride=stride, groups=groups, bias=False))
    res.add_module('bn', nn.BatchNorm2d(output_channel))
    return res


class wrapped_permute0231(nn.Module):
    def __init__(self):
        super(wrapped_permute0231, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class wrapped_permute0312(nn.Module):
    def __init__(self):
        super(wrapped_permute0312, self).__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


def _conv2d_ln(input_channel, output_channel, kernel_size=3, padding=1, stride=1, groups=1):
    res = nn.Sequential()
    res.add_module('conv', nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size,
                                     padding=padding, padding_mode='zeros', stride=stride, groups=groups, bias=False))
    res.add_module('trans1', wrapped_permute0231())
    res.add_module('norm', nn.LayerNorm(output_channel))
    res.add_module('trans2', wrapped_permute0312())
    return res


class RepConv2dNorm(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 bias: bool = True,
                 rep_list: Union[int, List[int]] = 1,
                 rep_alpha: Optional[Union[float, List[float]]] = None,
                 deploy: bool = False):
        super(RepConv2dNorm, self).__init__()
        assert padding in ((kernel_size - 1) // 2, 0), "Only support SAME/ZERO padding now."
        assert kernel_size % 2 == 1
        self.main_branch = _conv2d_bn(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        if type(rep_list) == int:
            rep_list = (rep_list,)
        self.rep_list = rep_list
        self.residual = nn.ModuleDict()
        self.with_alpha = False
        if rep_alpha is not None:
            if type(rep_alpha) == float:
                rep_alpha = (rep_alpha,)
                self.with_alpha = True
            self.rep_alpha = rep_alpha
            assert len(self.rep_alpha) == len(self.rep_list)
        for k in rep_list:
            assert k < kernel_size
            if k == 0:  # stands for shortcut branch
                self.residual['repconv{}'.format(k)] = _conv2d_bn(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
                self.residual.repconv0.conv.weight.data = torch.ones([out_channels, in_channels // groups, 1, 1])
                for param in self.residual.repconv0.conv.parameters():
                    param.requires_grad = False
            elif k % 2 == 0:  # even
                if padding == 0:
                    self.residual['repconv{}'.format(k)] = _conv2d_bn(
                        in_channels, out_channels, kernel_size=k, stride=stride, padding=0, groups=groups)
                else:
                    self.residual['repconv{}'.format(k)] = _pad_conv2d_bn(
                        in_channels, out_channels, kernel_size=k, stride=stride, groups=groups)
            else:
                branch_padding = (k - 1) // 2
                if padding == 0:
                    branch_padding = 0
                self.residual['repconv{}'.format(k)] = _conv2d_bn(
                    in_channels, out_channels, kernel_size=k, stride=stride, padding=branch_padding, groups=groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.deploy = deploy
        if not self.deploy:
            self.init_weights()

    def init_weights(self):
        conv_max = self.kernel_size ** -0.5
        torch.nn.init.uniform_(self.main_branch.conv.weight, -conv_max, conv_max)
        for k in self.rep_list:
            conv_max = max(k, 1) ** -0.5
            torch.nn.init.uniform_(self.residual.__getattr__('repconv{}'.format(k)).conv.weight, -conv_max, conv_max)

    def switch_to_deploy(self):
        self.deploy = True
        self.convmix = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, groups=self.groups, bias=True)
        mix_weight, mix_bias = self._fuse_conv_bn(self.main_branch)
        for idx, k in enumerate(self.rep_list):
            sub_weight, sub_bias = self._fuse_conv_bn(self.residual.__getattr__('repconv{}'.format(k)))
            if self.with_alpha:
                mix_weight += self.rep_alpha[idx] * self._pad_nxn_kernel(sub_weight, self.kernel_size, k)
                mix_bias += self.rep_alpha[idx] * sub_bias
            else:
                mix_weight += self._pad_nxn_kernel(sub_weight, self.kernel_size, k)
                mix_bias += sub_bias
        self.convmix.weight.data = mix_weight
        self.convmix.bias.data = mix_bias
        self.__delattr__('residual')
        self.__delattr__('main_branch')

    def _fuse_conv_bn(self, branch):
        if (branch is None):
            return 0, 0
        elif (isinstance(branch, nn.Sequential)):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.input_channel // self.groups
                kernel_value = np.zeros((self.input_channel, input_dim, self.kernel_size, self.kernel_size),
                                        dtype=np.float32)
                for i in range(self.input_channel):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = gamma / std
        t = t.view(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _pad_nxn_kernel(self, kernel: Optional[Tensor], kernel_size: int, k: int = 0):
        if kernel is None:
            return 0
        elif k != 0 and k % 2 == 0:
            up_l_pad = (kernel_size - k) // 2
            dn_r_pad = up_l_pad + 1
            return F.pad(kernel, [up_l_pad, dn_r_pad, up_l_pad, dn_r_pad])
        else:
            r = (kernel_size - k) // 2
            return F.pad(kernel, [r, r, r, r])

    def forward(self, inputs: Tensor) -> Tensor:
        if self.deploy:
            return self.convmix(inputs)
        else:
            outputs = self.main_branch(inputs)
            for idx, (_, repconv) in enumerate(self.residual.items()):
                if self.padding == 0:
                    k = self.rep_list[idx]
                    if k != 0 and k % 2 == 0:
                        ml = (self.kernel_size - k) // 2
                        mr = ml + 1
                        if self.with_alpha:
                            outputs += self.rep_alpha[idx] * repconv(inputs[:, :, ml:-mr, ml:-mr])
                        else:
                            outputs += repconv(inputs[:, :, ml:-mr, ml:-mr])
                    else:
                        m = (self.kernel_size - k) // 2
                        if self.with_alpha:
                            outputs += self.rep_alpha[idx] * repconv(inputs[:, :, m:-m, m:-m])
                        else:
                            outputs += repconv(inputs[:, :, m:-m, m:-m])
                else:
                    if self.with_alpha:
                        outputs += self.rep_alpha[idx] * repconv(inputs)
                    else:
                        outputs += repconv(inputs)
            return outputs


class DepthwiseRepConv2dNorm(RepConv2dNorm):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: int = 2,
            padding: Union[int, str] = 0,
            bias: bool = True,
            rep_list: Union[int, List[int]] = 1,
            rep_alpha: Optional[Union[float, List[float]]] = None,
            deploy: bool = False
    ) -> None:
        super(DepthwiseRepConv2dNorm, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=bias, rep_list=rep_list, rep_alpha=rep_alpha, deploy=deploy
        )


class RepConv2dBranch(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,  # main kernel size
            k: int,  # branch kernel size
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            bias: bool = True,
    ) -> None:
        super(RepConv2dBranch, self).__init__()
        self.kernel_size = kernel_size
        self.k = k
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        if kernel_size == k:
            self.branch = _conv2d_bn(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        elif k == 0:  # stands for shortcut branch
            self.branch = _conv2d_bn(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
            self.branch.conv.weight.data = torch.ones([out_channels, in_channels // groups, 1, 1])
            for param in self.branch.conv.parameters():
                param.requires_grad = False
        elif k % 2 == 0:  # even
            if padding == 0:
                self.branch = _conv2d_bn(
                    in_channels, out_channels, kernel_size=k, stride=stride, padding=0, groups=groups)
            else:
                self.branch = _pad_conv2d_bn(
                    in_channels, out_channels, kernel_size=k, stride=stride, groups=groups)
        else:
            branch_padding = (k - 1) // 2
            if padding == 0:
                branch_padding = 0
            self.branch = _conv2d_bn(
                in_channels, out_channels, kernel_size=k, stride=stride, padding=branch_padding, groups=groups)
        self.init_weights()

    def init_weights(self):
        conv_max = max(self.k, 1) ** -0.5
        torch.nn.init.uniform_(self.branch.conv.weight, -conv_max, conv_max)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.k == self.kernel_size:
            outputs = self.branch(inputs)
        elif self.padding == 0:
            if self.k != 0 and self.k % 2 == 0:
                ml = (self.kernel_size - self.k) // 2
                mr = ml + 1
                outputs = self.branch(inputs[:, :, ml:-mr, ml:-mr])
            else:
                m = (self.kernel_size - self.k) // 2
                outputs = self.branch(inputs[:, :, m:-m, m:-m])
        else:
            outputs = self.branch(inputs)
        return outputs


class DepthwiseRepConv2dBranch(RepConv2dBranch):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,  # main kernel size
                 k: int,  # branch kernel size
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True):
        super(DepthwiseRepConv2dBranch, self).__init__(
            in_channels, out_channels, kernel_size, k, stride, padding, in_channels, bias)


if __name__ == '__main__':
    ######### RepConv1d test #########
    # rep_conv1d_norm = RepConv1dNorm(in_channels=256, out_channels=256, kernel_size=31,
    #                                 stride=2, padding=(31 - 1) // 2, bias=True,
    #                                 rep_list=[0], deploy=False)
    rep_conv1d_norm = RepConv1dNorm(in_channels=256, out_channels=256, kernel_size=31,
                                    stride=3, padding=0, bias=True, groups=256,
                                    rep_list=[3, 4, 5, 2, 6], rep_alpha=[0.546, 0.291, 0.098, 0.034, 0.031],
                                    deploy=False)
    x = torch.rand(1, 256, 40)
    rep_conv1d_norm.eval()
    out1 = rep_conv1d_norm(x)
    print('out1', out1.size())
    rep_conv1d_norm.switch_to_deploy()
    out2 = rep_conv1d_norm(x)
    print('out2', out2.size())
    error = np.mean(np.abs(out1.detach().numpy() - out2.detach().numpy()))
    if error < 1e-4:
        print(termcolor.colored("PASS", color="green"), error)
    else:
        print(termcolor.colored("FAIL", color="red"), error)

    ######### RepConv2d test #########
    rep_conv2d_norm = RepConv2dNorm(in_channels=256, out_channels=256, kernel_size=31,
                                    stride=1, padding=(31 - 1) // 2, groups=256, bias=True,
                                    rep_list=[3, 4, 5, 2, 6], rep_alpha=[0.546, 0.291, 0.098, 0.034, 0.031],
                                    deploy=False)
    x = torch.rand(1, 256, 40, 40)
    rep_conv2d_norm.eval()
    out1 = rep_conv2d_norm(x)
    rep_conv2d_norm.switch_to_deploy()
    out2 = rep_conv2d_norm(x)
    error = np.mean(np.abs(out1.detach().numpy() - out2.detach().numpy()))
    if error < 1e-4:
        print(termcolor.colored("PASS", color="green"), error)
    else:
        print(termcolor.colored("FAIL", color="red"), error)

    ######### Conv1d branch test #########
    conv1d_branch = RepConv1dBranch(in_channels=256, out_channels=256, kernel_size=31, k=5,
                                    stride=1, padding=0, groups=256, bias=True)
    x = torch.ones([1, 256, 88])
    out = conv1d_branch(x)
    print('out', out.size())

    ######### Conv2d branch test #########
    conv2d_branch = RepConv2dBranch(in_channels=256, out_channels=256, kernel_size=31, k=2,
                                    stride=1, padding=(31 - 1) // 2, groups=256, bias=True)
    x = torch.ones([1, 256, 88, 20])
    out = conv2d_branch(x)
    print('out', out.size())
