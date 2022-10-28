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
# Modified from nni(https://github.com/microsoft/nni)

import copy
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

class LayerChoice(object):
    def __init__(self, candidates: Union[Dict[str, nn.Module], List[nn.Module]], label: Optional[str] = None):
        assert label is not None
        self.label = label
        self.names = []
        self.candidates = candidates
        self._modules = OrderedDict()
        if isinstance(candidates, dict):
            for name, module in candidates.items():
                assert name not in ["length", "reduction", "return_mask", "_key", "key", "names"], \
                    "Please don't use a reserved name '{}' for your module.".format(name)
                self.add_module(name, module)
                self.names.append(name)
        elif isinstance(candidates, list):
            for i, module in enumerate(candidates):
                self.add_module(str(i), module)
                self.names.append(str(i))
        else:
            raise TypeError("Unsupported candidates type: {}".format(type(candidates)))

    def add_module(self, name: str, module: Optional[nn.Module]) -> None:
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, nn.Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch.typename(module)))
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("module name should be a string. Got {}".format(
                torch.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return map(lambda name: self._modules[name], self.names)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._modules[idx]
        return list(self)[idx]

    def __setitem__(self, idx, module):
        key = idx if isinstance(idx, str) else self.names[idx]
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self.names[idx]:
                delattr(self, key)
        else:
            if isinstance(idx, str):
                key, idx = idx, self.names.index(idx)
            else:
                key = self.names[idx]
            delattr(self, key)
        del self.names[idx]


class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.op_alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3, requires_grad=True)

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.op_alpha, -1).view(*alpha_shape), 0)

    def parameters(self, **kwargs):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, recurse=False, **kwargs):
        for name, p in super(DartsLayerChoice, self).named_parameters(recurse=recurse):
            # if name == 'alpha':
            #     continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.op_alpha).item()]


class DartsLayerChoiceV2(nn.Module):
    def __init__(self, layer_choice, op_alpha):
        super(DartsLayerChoiceV2, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.op_alpha = op_alpha

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.op_alpha, -1).view(*alpha_shape), 0)

    def parameters(self, **kwargs):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, recurse=False, **kwargs):
        for name, p in super(DartsLayerChoiceV2, self).named_parameters(recurse=recurse):
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.op_alpha).item()]


class InputChoice(object):
    def __init__(self, n_candidates: int, n_chosen: Optional[int] = 1, label: Optional[str] = None):
        assert label is not None
        self.n_candidates = n_candidates
        self.n_chosen = n_chosen


class DartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.label
        self.op_alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.op_alpha, -1).view(*alpha_shape), 0)

    def parameters(self, **kwargs):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, recurse=False, **kwargs):
        for name, p in super(DartsInputChoice, self).named_parameters(recurse=recurse):
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.op_alpha).cpu().numpy().tolist()[:self.n_chosen]


class DoubleBranch(nn.Module):
    def __init__(self, main_branch, search_branches):
        super(DoubleBranch, self).__init__()
        self.main_branch = main_branch
        self.search_branches = search_branches

    def forward(self, inputs):
        outputs = self.main_branch(inputs) # depthwise branch
        outputs += self.search_branches(inputs) # search branches
        return outputs