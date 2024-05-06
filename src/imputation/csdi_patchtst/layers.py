"""
Modules for CSDI model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pypots.nn.modules.patchtst import PatchEmbedding, PatchtstEncoder, PredictionHead

def conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class CsdiDiffusionEmbedding(nn.Module):
    def __init__(self, n_diffusion_steps, d_embedding=128, d_projection=None):
        super().__init__()
        if d_projection is None:
            d_projection = d_embedding
        self.register_buffer(
            "embedding",
            self._build_embedding(n_diffusion_steps, d_embedding // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(d_embedding, d_projection)
        self.projection2 = nn.Linear(d_projection, d_projection)

    @staticmethod
    def _build_embedding(n_steps, d_embedding=64):
        steps = torch.arange(n_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (
            torch.arange(d_embedding) / (d_embedding - 1) * 4.0
        ).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

    def forward(self, diffusion_step: int):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

class CsdiDiffusionModel_PatchTST(nn.Module):
    def __init__(
        self,
        n_diffusion_steps,
        n_steps,
        patch_len,
        stride,
        d_diffusion_embedding,
        d_model,
        d_ffn,
        d_k,
        d_v,
        d_side,
        n_features,
        n_heads,
        n_layers,
    ):
        super().__init__()
        self.diffusion_embedding = CsdiDiffusionEmbedding(
            n_diffusion_steps=n_diffusion_steps,
            d_embedding=d_diffusion_embedding,
        )

        self.cond_projection = conv1d_with_init(d_side * n_features, d_model, 1)
        self.x_projection = conv1d_with_init(2 * n_features, d_model, 1)
        self.diffusion_projection = nn.Linear(d_diffusion_embedding, d_model)
        self.output_projection = nn.Linear(d_model, n_features)
        padding = stride
        n_patches = int((n_steps - patch_len) / stride + 2)  # number of patches

        self.d_model = d_model
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, 0
        )
        self.encoder = PatchtstEncoder(
            n_layers, n_heads, d_model, d_ffn, d_k, d_v, 0, 0
        )
        self.head = PredictionHead(d_model, n_patches, n_steps, 0)

    def forward(self, x, cond_info, diffusion_step):
        (
            n_samples,
            input_dim,
            n_features,
            n_steps,
        ) = x.shape  # n_samples, 2, n_features, n_steps
        _, cond_dim, _, _ = cond_info.shape

        x = x.reshape(n_samples, input_dim * n_features, n_steps)
        x = self.x_projection(x)  # n_samples, d_model, n_steps
        x = F.relu(x)

        cond_info = cond_info.reshape(n_samples, cond_dim * n_features, n_steps)
        cond_info = self.cond_projection(cond_info)
        cond_info = F.relu(cond_info)

        x = x + cond_info

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        x = x + self.diffusion_projection(diffusion_emb).unsqueeze(-1)

        enc_in = self.patch_embedding(x)
        enc_out, attns = self.encoder(enc_in)
        x = self.head(enc_out)
        x = self.output_projection(x)

        return x.permute(0, 2, 1)
