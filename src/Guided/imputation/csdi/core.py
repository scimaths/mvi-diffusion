"""
The core wrapper assembles the submodules of CSDI imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules.csdi import BackboneCSDI, BackboneCSDI_PatchTST


class _CSDI(nn.Module):
    def __init__(
        self,
        n_features,
        n_layers,
        n_heads,
        n_channels,
        d_time_embedding,
        d_feature_embedding,
        d_diffusion_embedding,
        is_unconditional,
        n_diffusion_steps,
        schedule,
        beta_start,
        beta_end,
        cls_free = False,
        drop_prob_tot = 0.1,
        drop_prob_iid = 0.2,
        guidance = 0.1,
        use_patchtst  = False,
        n_steps = None,
        patch_len = None,
        stride = None,
        d_model = None,
        d_ffn = None ,
        d_k = None,
        d_v = None

    ):
        super().__init__()

        self.n_features = n_features
        self.d_time_embedding = d_time_embedding
        self.is_unconditional = is_unconditional
        self.drop_prob_iid = drop_prob_iid
        self.drop_prob_tot = drop_prob_tot
        self.patchtst = use_patchtst

        self.embed_layer = nn.Embedding(
            num_embeddings=n_features,
            embedding_dim=d_feature_embedding,
        )
        self.backbone = BackboneCSDI(
            n_features,
            n_layers,
            n_heads,
            n_channels,
            n_features,
            d_time_embedding,
            d_feature_embedding,
            d_diffusion_embedding,
            is_unconditional,
            n_diffusion_steps,
            schedule,
            beta_start,
            beta_end,
            cls_free,
            drop_prob_tot,
            drop_prob_iid,
            guidance
        )

        if self.patchtst:
            self.backbone = BackboneCSDI_PatchTST(
            n_steps,
            patch_len,
            stride,
            n_layers,
            n_heads,
            n_features,
            d_model,
            d_ffn,
            d_k,
            d_v,
            d_time_embedding,
            d_feature_embedding,
            d_diffusion_embedding,
            is_unconditional,
            n_diffusion_steps,
            schedule,
            beta_start,
            beta_end,
        )

    @staticmethod
    def time_embedding(pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(pos.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=pos.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    

    def forward(self, inputs, training=True, n_sampling_times=1):
        results = {}
        if training:  # for training
            (observed_data, indicating_mask, cond_mask, observed_tp) = (
                inputs["X_ori"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            #side_info = self.get_side_info(observed_tp, cond_mask)
            #cond_mask = torch.bernoulli(torch.zeros_like(cond_mask)+(1-self.drop_prob_iid)).to(cond_mask.device)
            training_loss = self.backbone.calc_loss(
                observed_data, cond_mask, indicating_mask, observed_tp, training
            )
            results["loss"] = training_loss
        elif not training and n_sampling_times == 0:  # for validating
            (observed_data, indicating_mask, cond_mask, observed_tp) = (
                inputs["X_ori"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            #side_info = self.get_side_info(observed_tp, cond_mask)
            validating_loss = self.backbone.calc_loss_valid(
                observed_data, cond_mask, indicating_mask, observed_tp, training
            )
            results["loss"] = validating_loss
        elif not training and n_sampling_times > 0:  # for testing
            observed_data, cond_mask, observed_tp = (
                inputs["X"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            #side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.backbone(
                observed_data, cond_mask, observed_tp, n_sampling_times
            )  # (n_samples, n_sampling_times, n_features, n_steps)
            repeated_obs = observed_data.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            repeated_mask = cond_mask.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            imputed_data = repeated_obs + samples * (1 - repeated_mask)

            results["imputed_data"] = imputed_data.permute(
                0, 1, 3, 2
            )  # (n_samples, n_sampling_times, n_steps, n_features)

        return results
