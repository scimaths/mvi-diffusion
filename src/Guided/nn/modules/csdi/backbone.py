"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn

from .layers import CsdiDiffusionModel, CsdiDiffusionModel_PatchTST

#data pipeline changed for classifier guidance


class BackboneCSDI(nn.Module):
    def __init__(
        self,
        n_features,
        n_layers,
        n_heads,
        n_channels,
        d_target,
        d_time_embedding,
        d_feature_embedding,
        d_diffusion_embedding,
        is_unconditional,
        n_diffusion_steps,
        schedule,
        beta_start,
        beta_end,
        cls_free = False, #if classifier free
        drop_prob_tot = 0.1, #prob to drop entire cond mask in training
        drop_prob_iid = 0.2, #prob to drop conditional mask entries in training
        guidance = 1.0,  #strength of generative guidance
        
    ):
        super().__init__()

        self.n_features = n_features
        self.d_target = d_target
        self.d_time_embedding = d_time_embedding
        self.d_feature_embedding = d_feature_embedding
        self.is_unconditional = is_unconditional
        self.n_channels = n_channels
        self.n_diffusion_steps = n_diffusion_steps

        #setting classifier guidance config
        self.cls_free = cls_free
        self.drop_prob_tot = drop_prob_tot
        self.drop_prob_iid = drop_prob_iid
        self.guidance = guidance

        d_side = d_time_embedding + d_feature_embedding
        if self.is_unconditional:
            d_input = 1
        else:
            d_side += 1  # for conditional mask
            d_input = 2

        self.diff_model = CsdiDiffusionModel(
            n_diffusion_steps,
            d_diffusion_embedding,
            d_input,
            d_side,
            n_channels,
            n_heads,
            n_layers,
        )

        self.embed_layer = nn.Embedding(
            num_embeddings=n_features,
            embedding_dim=d_feature_embedding,
        )

        if cls_free:
            print("USING CLASSIFIER FREE GUIDANCE")
        # parameters for diffusion models
        if schedule == "quad":
            self.beta = (
                np.linspace(beta_start**0.5, beta_end**0.5, self.n_diffusion_steps)
                ** 2
            )
        elif schedule == "linear":
            self.beta = np.linspace(beta_start, beta_end, self.n_diffusion_steps)
        else:
            raise ValueError(
                f"The argument schedule should be 'quad' or 'linear', but got {schedule}"
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer(
            "alpha_torch", torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)
        )

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        device = observed_tp.device
        time_embed = self.time_embedding(
            observed_tp, self.d_time_embedding
        )  # (B,L,emb)
        time_embed = time_embed.to(device)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.n_features).to(device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat(
            [time_embed, feature_embed], dim=-1
        )  # (B,L,K,emb+d_feature_embedding)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info
    
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

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def calc_loss_valid(
        self, observed_data, cond_mask, indicating_mask, observed_tp ,  is_train
    ):
        loss_sum = 0
        for t in range(self.n_diffusion_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, indicating_mask, observed_tp, is_train,  set_t=t, drop_cond = False
            )
            loss_sum += loss.detach()
        return loss_sum / self.n_diffusion_steps

    def calc_loss(
        self, observed_data, cond_mask, indicating_mask, observed_tp, is_train,  set_t=-1, drop_cond = True
    ):
        B, K, L = observed_data.shape
        device = observed_data.device
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(device)
        else:
            t = torch.randint(0, self.n_diffusion_steps, [B]).to(device)

        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (
            1.0 - current_alpha
        ) ** 0.5 * noise

        #drop conditionfor unconditional generation in classifier free guidance
        """
        if self.cls_free and drop_cond:
            #cond_mask = torch.bernoulli(torch.zeros_like(cond_mask)+(1-self.drop_prob_iid)).to(device)
            #cond_mask = cond_mask * torch.bernoulli(torch.tensor(1-self.drop_prob_tot)).to(device)
            pass
        """

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        predicted = self.diff_model(total_input, side_info, t)  # (B,K,L)

        target_mask = indicating_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        loss = loss/5 if self.cls_free else loss
        return loss

    def forward(self, observed_data, cond_mask, observed_tp, n_sampling_times, clip_guide = None, clip_preproc = None, clip_guidance = 1.0):
        B, K, L = observed_data.shape
        device = observed_data.device
        imputed_samples = torch.zeros(B, n_sampling_times, K, L).to(device)
        side_info = self.get_side_info(observed_tp, cond_mask)
        print("n sampling ", n_sampling_times)

        for i in range(n_sampling_times):
            # generate noisy observation for unconditional model
            if not self.cls_free and clip_guide is None:
                if self.is_unconditional:
                    noisy_obs = observed_data
                    noisy_cond_history = []
                    for t in range(self.n_diffusion_steps):
                        noise = torch.randn_like(noisy_obs)
                        noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                            t
                        ] ** 0.5 * noise
                        noisy_cond_history.append(noisy_obs * cond_mask)

                current_sample = torch.randn_like(observed_data)

                for t in range(self.n_diffusion_steps - 1, -1, -1):
                    if self.is_unconditional:
                        diff_input = (
                            cond_mask * noisy_cond_history[t]
                            + (1.0 - cond_mask) * current_sample
                        )
                        diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    predicted = self.diff_model(
                        diff_input, side_info, torch.tensor([t]).to(device)
                    )

                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)

                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (
                            (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                        ) ** 0.5
                        current_sample += sigma * noise

                imputed_samples[:, i] = current_sample.detach()
            # CLASSIFIER FREE GUIDANCE
            elif clip_guide is None:
                current_sample = torch.randn_like(observed_data)
                for t in range(self.n_diffusion_steps-1, -1, -1):
                    
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input_cond = torch.cat([cond_obs, noisy_target], dim = 1)

                    cond_null_obs = (torch.zeros_like(cond_mask) * observed_data).unsqueeze(1)
                    side_info_null = self.get_side_info(observed_tp, torch.zeros_like(cond_mask))
                    noisy_target_null = current_sample.unsqueeze(1)
                    diff_input_null = torch.cat([cond_null_obs, noisy_target_null], dim = 1)

                    eps_pred_cond = self.diff_model(diff_input_cond, side_info, torch.tensor([t]).to(device))
                    eps_pred_null = self.diff_model(diff_input_null, side_info_null, torch.tensor([t]).to(device))
                    eps_pred_cls_guide = (1 + self.guidance) * eps_pred_cond - self.guidance * eps_pred_null
                    sampled_noise = torch.randn_like(current_sample)
                    scaling = (1 / self.alpha_hat[t]) ** 0.5
                    shift_ratio = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    shift = shift_ratio * eps_pred_cls_guide
                    mean = scaling * (current_sample - shift)
                    sigma = ((1 - self.alpha[t-1]) / (1 - self.alpha[t]) * self.beta[t]) ** 0.5 if t > 0 else 0
                    current_sample = mean + sigma * sampled_noise

                imputed_samples[:, i] = current_sample.detach()
            #CLIP Guidance
            else:
                for t in range(self.n_diffusion_steps - 1, -1, -1):
                    if self.is_unconditional:
                        diff_input = (
                            cond_mask * noisy_cond_history[t]
                            + (1.0 - cond_mask) * current_sample
                        )
                        diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    predicted = self.diff_model(
                        diff_input, side_info, torch.tensor([t]).to(device)
                    )

                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)

                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (
                            (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                        ) ** 0.5
                        current_sample += sigma * noise
                        preproc_module = torch.cat((current_sample, observed_data), dim = 0)
                        clip_module = clip_preproc(preproc_module)
                        shift = sigma * clip_guide(clip_module)
                        current_sample += clip_guidance*shift

                imputed_samples[:, i] = current_sample.detach()


        return imputed_samples
    

class BackboneCSDI_CLIP(nn.Module):
    def __init__(
            self,
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
        ):
            super().__init__()

            self.n_features = n_features
            self.d_time_embedding = d_time_embedding
            self.d_feature_embedding = d_feature_embedding
            self.is_unconditional = is_unconditional
            self.n_diffusion_steps = n_diffusion_steps

            d_side = d_time_embedding + d_feature_embedding
            if not self.is_unconditional:
                d_side += 1  # for conditional mask

            self.embed_layer = nn.Embedding(
                num_embeddings=n_features,
                embedding_dim=d_feature_embedding,
            )

            self.diff_model = CsdiDiffusionModel_PatchTST(
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
            )

            # parameters for diffusion models
            if schedule == "quad":
                self.beta = (
                    np.linspace(beta_start**0.5, beta_end**0.5, self.n_diffusion_steps)
                    ** 2
                )
            elif schedule == "linear":
                self.beta = np.linspace(beta_start, beta_end, self.n_diffusion_steps)
            else:
                raise ValueError(
                    f"The argument schedule should be 'quad' or 'linear', but got {schedule}"
                )

            self.alpha_hat = 1 - self.beta
            self.alpha = np.cumprod(self.alpha_hat)
            self.register_buffer(
                "alpha_torch", torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)
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

    def get_side_info(self, observed_tp, cond_mask):
            B, K, L = cond_mask.shape
            device = observed_tp.device
            time_embed = self.time_embedding(
                observed_tp, self.d_time_embedding
            )  # (B,L,emb)
            time_embed = time_embed.to(device)
            time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
            feature_embed = self.embed_layer(
                torch.arange(self.n_features).to(device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

            side_info = torch.cat(
                [time_embed, feature_embed], dim=-1
            )  # (B,L,K,emb+d_feature_embedding)
            side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

            if not self.is_unconditional:
                side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
                side_info = torch.cat([side_info, side_mask], dim=1)

            return side_info

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
            if self.is_unconditional:
                total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
            else:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

            return total_input

    def calc_loss_valid(
            self, observed_data, cond_mask, indicating_mask, side_info, is_train
        ):
            loss_sum = 0
            for t in range(self.n_diffusion_steps):  # calculate loss for all t
                loss = self.calc_loss(
                    observed_data, cond_mask, indicating_mask, side_info, is_train, set_t=t
                )
                loss_sum += loss.detach()
            return loss_sum / self.n_diffusion_steps

    def calc_loss(
            self, observed_data, cond_mask, indicating_mask, side_info, is_train, set_t=-1
        ):
            B, K, L = observed_data.shape
            device = observed_data.device
            if is_train != 1:  # for validation
                t = (torch.ones(B) * set_t).long().to(device)
            else:
                t = torch.randint(0, self.n_diffusion_steps, [B]).to(device)

            current_alpha = self.alpha_torch[t]  # (B,1,1)
            noise = torch.randn_like(observed_data)
            noisy_data = (current_alpha**0.5) * observed_data + (
                1.0 - current_alpha
            ) ** 0.5 * noise

            total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

            predicted = self.diff_model(total_input, side_info, t)  # (B,K,L)

            target_mask = indicating_mask
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
            return loss

    def forward(self, observed_data, cond_mask, side_info, n_sampling_times, feature_fn):
            B, K, L = observed_data.shape
            device = observed_data.device
            imputed_samples = torch.zeros(B, n_sampling_times, K, L).to(device)

            for i in range(n_sampling_times):
                # generate noisy observation for unconditional model
                if self.is_unconditional:
                    noisy_obs = observed_data
                    noisy_cond_history = []
                    for t in range(self.n_diffusion_steps):
                        noise = torch.randn_like(noisy_obs)
                        noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                            t
                        ] ** 0.5 * noise
                        noisy_cond_history.append(noisy_obs * cond_mask)

                current_sample = torch.randn_like(observed_data)

                for t in range(self.n_diffusion_steps - 1, -1, -1):
                    if self.is_unconditional:
                        diff_input = (
                            cond_mask * noisy_cond_history[t]
                            + (1.0 - cond_mask) * current_sample
                        )
                        diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    predicted = self.diff_model(
                        diff_input, side_info, torch.tensor([t]).to(device)
                    )

                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)

                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (
                            (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                        ) ** 0.5
                        current_sample += sigma * noise
                        current_sample -= feature_fn(current_sample)
                        

                imputed_samples[:, i] = current_sample.detach()
            return imputed_samples

    #CSDI patchTST
class BackboneCSDI_PatchTST(nn.Module):
        def __init__(
            self,
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
        ):
            super().__init__()

            self.n_features = n_features
            self.d_time_embedding = d_time_embedding
            self.d_feature_embedding = d_feature_embedding
            self.is_unconditional = is_unconditional
            self.n_diffusion_steps = n_diffusion_steps

            d_side = d_time_embedding + d_feature_embedding
            if not self.is_unconditional:
                d_side += 1  # for conditional mask

            self.embed_layer = nn.Embedding(
                num_embeddings=n_features,
                embedding_dim=d_feature_embedding,
            )

            self.diff_model = CsdiDiffusionModel_PatchTST(
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
            )

            # parameters for diffusion models
            if schedule == "quad":
                self.beta = (
                    np.linspace(beta_start**0.5, beta_end**0.5, self.n_diffusion_steps)
                    ** 2
                )
            elif schedule == "linear":
                self.beta = np.linspace(beta_start, beta_end, self.n_diffusion_steps)
            else:
                raise ValueError(
                    f"The argument schedule should be 'quad' or 'linear', but got {schedule}"
                )

            self.alpha_hat = 1 - self.beta
            self.alpha = np.cumprod(self.alpha_hat)
            self.register_buffer(
                "alpha_torch", torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)
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

        def get_side_info(self, observed_tp, cond_mask):
            B, K, L = cond_mask.shape
            device = observed_tp.device
            time_embed = self.time_embedding(
                observed_tp, self.d_time_embedding
            )  # (B,L,emb)
            time_embed = time_embed.to(device)
            time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
            feature_embed = self.embed_layer(
                torch.arange(self.n_features).to(device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

            side_info = torch.cat(
                [time_embed, feature_embed], dim=-1
            )  # (B,L,K,emb+d_feature_embedding)
            side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

            if not self.is_unconditional:
                side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
                side_info = torch.cat([side_info, side_mask], dim=1)

            return side_info

        def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
            if self.is_unconditional:
                total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
            else:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

            return total_input

        def calc_loss_valid(
            self, observed_data, cond_mask, indicating_mask, side_info, is_train
        ):
            loss_sum = 0
            for t in range(self.n_diffusion_steps):  # calculate loss for all t
                loss = self.calc_loss(
                    observed_data, cond_mask, indicating_mask, side_info, is_train, set_t=t
                )
                loss_sum += loss.detach()
            return loss_sum / self.n_diffusion_steps

        def calc_loss(
            self, observed_data, cond_mask, indicating_mask, side_info, is_train, set_t=-1
        ):
            B, K, L = observed_data.shape
            device = observed_data.device
            if is_train != 1:  # for validation
                t = (torch.ones(B) * set_t).long().to(device)
            else:
                t = torch.randint(0, self.n_diffusion_steps, [B]).to(device)

            current_alpha = self.alpha_torch[t]  # (B,1,1)
            noise = torch.randn_like(observed_data)
            noisy_data = (current_alpha**0.5) * observed_data + (
                1.0 - current_alpha
            ) ** 0.5 * noise

            total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

            predicted = self.diff_model(total_input, side_info, t)  # (B,K,L)

            target_mask = indicating_mask
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
            return loss

        def forward(self, observed_data, cond_mask, side_info, n_sampling_times):
            B, K, L = observed_data.shape
            device = observed_data.device
            imputed_samples = torch.zeros(B, n_sampling_times, K, L).to(device)

            for i in range(n_sampling_times):
                # generate noisy observation for unconditional model
                if self.is_unconditional:
                    noisy_obs = observed_data
                    noisy_cond_history = []
                    for t in range(self.n_diffusion_steps):
                        noise = torch.randn_like(noisy_obs)
                        noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                            t
                        ] ** 0.5 * noise
                        noisy_cond_history.append(noisy_obs * cond_mask)

                current_sample = torch.randn_like(observed_data)

                for t in range(self.n_diffusion_steps - 1, -1, -1):
                    if self.is_unconditional:
                        diff_input = (
                            cond_mask * noisy_cond_history[t]
                            + (1.0 - cond_mask) * current_sample
                        )
                        diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    predicted = self.diff_model(
                        diff_input, side_info, torch.tensor([t]).to(device)
                    )

                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)

                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (
                            (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                        ) ** 0.5
                        current_sample += sigma * noise

                imputed_samples[:, i] = current_sample.detach()
            return imputed_samples
