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
# Modified from rotary-embedding-torch(https://github.com/lucidrains/rotary-embedding-torch)

"""RotaryEmbedding definition."""

import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
    ):
        super().__init__()
        if custom_freqs is not None:
            self.freqs = custom_freqs
        elif freqs_for == 'lang':
            self.freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            self.freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            self.freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()

    def repeat_torch(self, x, r: int = 2):
        x = x.unsqueeze(-1).repeat(1, 1, r)
        x = x.flatten(-2, -1)
        return x

    def rotate_half(self, x):
        b, t, d = x.size(0), x.size(1), x.size(2)
        assert d % 2 == 0
        x = x.contiguous().view(b, t, d//2, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2, -1)

    def apply_rotary_emb(self, freqs: torch.Tensor, t: torch.Tensor, start_index: int=0) -> torch.Tensor:
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[
            -1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * freqs.cos()) + (self.rotate_half(t) * freqs.sin())
        return torch.cat((t_left, t, t_right), dim=-1)

    def forward(self, x):
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device)
        freqs = self.freqs.to(x.device)
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = self.repeat_torch(freqs, r=2)
        x = self.apply_rotary_emb(freqs, x)
        return x

if __name__ == '__main__':
    rotary_emb = RotaryEmbedding(dim=128, freqs_for='lang')
    torch.jit.script(rotary_emb)
    q = torch.rand(1, 128, 256)
    qr = rotary_emb(q)
    print('qr', qr.size())