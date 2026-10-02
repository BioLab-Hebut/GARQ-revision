import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GARQuantizer(nn.Module):
    def __init__(self, entry_num, entry_dim, k_knn=5):
        super().__init__()
        self.entry_num = entry_num  # ancher num
        self.entry_dim = entry_dim  
        self.decay = 0.9  
        self.k_knn = k_knn  
        
        self.anchors = nn.Embedding(self.entry_num, self.entry_dim)
        self.register_buffer("anchor_usage", torch.zeros(self.entry_num))
        self.diagnostics_enabled = False
        self.dynamic_update_enabled = True
        self.reposition_interval = 1
        self.diagnostic_trace = []
        self.diagnostic_epoch = None
        self.diagnostic_phase = None
        self._quantized_training_step = 0

    def configure_anchor_diagnostics(
        self,
        enabled=False,
        dynamic_update_enabled=True,
        reposition_interval=1,
    ):
        if int(reposition_interval) < 1:
            raise ValueError("reposition_interval must be >= 1")
        self.diagnostics_enabled = bool(enabled)
        self.dynamic_update_enabled = bool(dynamic_update_enabled)
        self.reposition_interval = int(reposition_interval)
        self.diagnostic_trace = []
        self._quantized_training_step = 0

    def init_anchors(self, z, method):
        import faiss
        d = z.shape[1]
        kmeans = faiss.Kmeans(d, self.entry_num, spherical=True, gpu=True)
        kmeans.train(z.detach().cpu().numpy())
        D, I = kmeans.index.search(z.detach().cpu().numpy(), 1)
        assignments = I.reshape(-1)
        centers = np.zeros((self.entry_num, d))
        for i in range(self.entry_num):
            centers[i] = z.detach().cpu().numpy()[assignments == i].mean(axis=0)
        self.anchors.weight.data.copy_(torch.from_numpy(centers).to(z.device))


    def build_cell_knn_graph(self, e):

        batch_size = e.shape[0]
        if batch_size <= self.k_knn:
            return torch.ones(batch_size, batch_size, device=e.device) / batch_size
        
        normed_e = F.normalize(e, dim=1)
        cell_sim = torch.matmul(normed_e, normed_e.T)  
        
        topk_sim, topk_idx = torch.topk(cell_sim, k=self.k_knn + 1, dim=1)  
        topk_idx = topk_idx[:, 1:]  
        
        adj = torch.zeros_like(cell_sim, device=e.device)
        adj.scatter_(dim=1, index=topk_idx, value=1.0)
        
        adj = (adj + adj.T) / 2
        return adj

    def encode_relation(self, e):

        normed_e = F.normalize(e, dim=1).detach()
        normed_anchors = F.normalize(self.anchors.weight, dim=1)
        raw_sim = torch.einsum("bd,dn->bn", normed_e, rearrange(normed_anchors, "n d -> d n"))
        
        adj = self.build_cell_knn_graph(e)  
        smoothed_sim = torch.matmul(adj, raw_sim)  
        
        self.alpha = nn.Parameter(torch.tensor(0.2))  
        sim = raw_sim + self.alpha * smoothed_sim  
        sim = raw_sim + torch.clamp(self.alpha, min=0) * smoothed_sim
        
        top2_sim, top2_idx = torch.topk(sim, 2, dim=1)
        relations = torch.zeros_like(sim)
        for i in range(e.shape[0]):
            relations[i, top2_idx[i]] = 1.0
        return sim, relations

    def forward(self, e, return_assignment):
        sim, relations = self.encode_relation(e)

        assignment_indices = torch.argmax(sim, dim=1)
        assignments = torch.zeros(
            assignment_indices.unsqueeze(1).shape[0], self.entry_num, device=e.device
        )
        assignments.scatter_(1, assignment_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(assignments, dim=0)

        e_q = torch.matmul(assignments, self.anchors.weight)
        loss = torch.mean((e_q - e.detach()) ** 2)

        if self.training:
            self._quantized_training_step += 1
            anchors_before = (
                self.anchors.weight.detach().clone()
                if self.diagnostics_enabled
                else None
            )
            self.anchor_usage.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            reposition_due = (
                self.dynamic_update_enabled
                and self._quantized_training_step % self.reposition_interval == 0
            )
            beta_s = None
            beta_l = None
            local_branch_condition = bool(self.anchor_usage.sum() + 1e-4 >= 1)
            if reposition_due:
                norm_distance = F.softmax(1 - sim, dim=1)
                norm_distance = torch.max(norm_distance, dim=1).values
                dis_indices = torch.multinomial(
                    norm_distance, num_samples=self.entry_num, replacement=True
                ).view(-1)
                random_feat = e.detach()[dis_indices]
                beta_s = (
                    torch.exp(-self.anchor_usage * self.entry_num * 100 - 1e-3)
                    .unsqueeze(1)
                    .repeat(1, self.entry_dim)
                )
                self.anchors.weight.data = (
                    self.anchors.weight.data * (1 - beta_s) + random_feat * beta_s
                )

                if local_branch_condition:
                    sim_t = sim.t()
                    median_distance = torch.median(sim_t, dim=1).values
                    median_distance = torch.abs(sim_t - median_distance[:, None])
                    dis_indices = torch.multinomial(
                        F.softmax(-median_distance, dim=1), num_samples=1
                    ).view(-1)
                    random_feat = e.detach()[dis_indices]
                    beta_l = (
                        torch.exp(-self.anchor_usage.mean() / self.anchor_usage * 10 - 1e-3)
                        .unsqueeze(1)
                        .repeat(1, self.entry_dim)
                    )
                    self.anchors.weight.data = (
                        self.anchors.weight.data * (1 - beta_l) + random_feat * beta_l
                    )

            if self.diagnostics_enabled:
                usage = self.anchor_usage.detach()
                usage_total = usage.sum()
                probabilities = usage / usage_total if usage_total > 0 else usage
                positive = probabilities[probabilities > 0]
                entropy = (
                    -torch.sum(positive * torch.log(positive))
                    if len(positive)
                    else torch.tensor(0.0, device=usage.device)
                )
                ordered = torch.sort(usage).values
                index = torch.arange(1, self.entry_num + 1, device=usage.device)
                gini = (
                    torch.sum((2 * index - self.entry_num - 1) * ordered)
                    / (self.entry_num * usage_total)
                    if usage_total > 0
                    else torch.tensor(0.0, device=usage.device)
                )
                displacement = torch.linalg.vector_norm(
                    self.anchors.weight.detach() - anchors_before, dim=1
                )

                def beta_stats(value):
                    if value is None:
                        return (float("nan"), float("nan"), float("nan"))
                    flattened = value[:, 0].detach()
                    return (
                        float(flattened.min()),
                        float(flattened.median()),
                        float(flattened.max()),
                    )

                beta_s_min, beta_s_median, beta_s_max = beta_stats(beta_s)
                beta_l_min, beta_l_median, beta_l_max = beta_stats(beta_l)
                self.diagnostic_trace.append(
                    {
                        "training_step": self._quantized_training_step,
                        "epoch": self.diagnostic_epoch,
                        "phase": self.diagnostic_phase,
                        "batch_cells": int(e.shape[0]),
                        "assignment_nonempty_anchor_count": int(torch.sum(avg_probs > 0)),
                        "usage_sum": float(usage_total),
                        "usage_min": float(usage.min()),
                        "usage_median": float(usage.median()),
                        "usage_max": float(usage.max()),
                        "usage_entropy_nats": float(entropy),
                        "usage_perplexity": float(torch.exp(entropy)),
                        "usage_gini": float(gini),
                        "manual_reposition_enabled": self.dynamic_update_enabled,
                        "reposition_interval": self.reposition_interval,
                        "reposition_due": reposition_due,
                        "local_branch_condition": local_branch_condition,
                        "local_branch_executed": bool(reposition_due and local_branch_condition),
                        "beta_s_min": beta_s_min,
                        "beta_s_median": beta_s_median,
                        "beta_s_max": beta_s_max,
                        "beta_l_min": beta_l_min,
                        "beta_l_median": beta_l_median,
                        "beta_l_max": beta_l_max,
                        "manual_displacement_min": float(displacement.min()),
                        "manual_displacement_median": float(displacement.median()),
                        "manual_displacement_max": float(displacement.max()),
                        "manual_displacement_anchor_count": int(torch.sum(displacement > 0)),
                        "anchor_nan_count": int(torch.isnan(self.anchors.weight).sum()),
                        "anchor_inf_count": int(torch.isinf(self.anchors.weight).sum()),
                    }
                )

        if return_assignment:
            top2_sim = torch.topk(sim, 2, dim=1).values
            delta_conf = top2_sim[:, 0] - top2_sim[:, 1]
            loss_c_sum = torch.sum((e_q - e.detach()) ** 2)
            return assignment_indices, delta_conf, loss_c_sum
        else:
            return e_q, loss


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads=4, ff_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)  
        x = self.transformer(x)
        x = x.squeeze(1)  
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, entry_dim, output_dim, n_heads=4, ff_dim=256):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=entry_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.decoder_mean = nn.Sequential(
            nn.Linear(entry_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.decoder_disp = nn.Sequential(
            nn.Linear(entry_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Softplus()
        )
        self.decoder_pi = nn.Sequential(
            nn.Linear(entry_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        mean = torch.clamp(torch.exp(self.decoder_mean(x)), 1e-5, 1e6)
        disp = torch.clamp(self.decoder_disp(x), 1e-4, 1e4)
        pi = self.decoder_pi(x)  
        return mean, disp, pi

class Decoder_ATAC(nn.Module):
    def __init__(self, entry_dim, output_dim, n_heads=8, ff_dim=512):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(entry_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.mean_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.disp_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return (
            torch.clamp(torch.exp(self.mean_head(x)), 1e-5, 1e6),
            torch.clamp(self.disp_head(x), 1e-4, 1e4),
            None
        )
def get_decoder(data_type, entry_dim, output_dim):
    if data_type in ["RNA", "ADT"]:
        return TransformerDecoder(entry_dim, output_dim)
    elif data_type == "ATAC":
        return Decoder_ATAC(entry_dim, output_dim)



class GARQ(nn.Module):
    def __init__(
        self,
        input_dims,
        data_types,
        entry_num,
        entry_dim=32,
        k_knn=5,
        modality_weights=None,
    ):
        super(GARQ, self).__init__()
        self.omics_num = len(input_dims)
        if modality_weights is not None:
            if len(modality_weights) != self.omics_num:
                raise ValueError("modality_weights must match the number of modalities")
            if any(float(weight) < 0 for weight in modality_weights):
                raise ValueError("modality_weights must be non-negative")
            if not any(float(weight) > 0 for weight in modality_weights):
                raise ValueError("at least one modality weight must be positive")
            self.modality_weights = tuple(float(weight) for weight in modality_weights)
        else:
            self.modality_weights = None
        self.encoders = nn.ModuleList(
            [TransformerEncoder(input_dim, entry_dim) for input_dim in input_dims]
        )

        self.quantizer = GARQuantizer(entry_num, entry_dim * self.omics_num, k_knn=k_knn)
        self.decoders = nn.ModuleList(
            [
                get_decoder(data_type, entry_dim, input_dim)
                for input_dim, data_type in zip(input_dims, data_types)
            ]
        )
        self.decoders_q = nn.ModuleList(
            [
                get_decoder(data_type, entry_dim * self.omics_num, input_dim)
                for input_dim, data_type in zip(input_dims, data_types)
            ]
        )

    def init_quantizer(self, z, method="Kmeans"):
        self.quantizer.init_anchors(z, method)

    def copy_decoder_q(self):
        if self.omics_num == 1:
            self.decoders_q[0].load_state_dict(self.decoders[0].state_dict())

    def combine_hiddens(self, hiddens):
        if self.modality_weights is None:
            weighted = hiddens
        else:
            weighted = [
                hidden * weight
                for hidden, weight in zip(hiddens, self.modality_weights)
            ]
        if self.omics_num > 1:
            return torch.cat(weighted, dim=1)
        else:
            return weighted[0]

    def quantize(self, hiddens, return_assignment=False):
        hidden = self.combine_hiddens(hiddens)
        return self.quantizer(hidden, return_assignment)

    def forward(self, inputs):
        hiddens = []
        for i in range(self.omics_num):
            hiddens.append(self.encoders[i](inputs[i]))
        return hiddens

    def decode(self, hiddens):
        means, disps,pis = [], [],[]
        for i in range(self.omics_num):
            mean, disp,pi = self.decoders[i](hiddens[i])
            means.append(mean)
            disps.append(disp)
            pis.append(pi)

        return means, disps,pis

    def decode_q(self, hidden):
        means, disps,pis = [], [],[]
        for i in range(self.omics_num):
            mean, disp,pi = self.decoders_q[i](hidden)
            means.append(mean)
            disps.append(disp)
            pis.append(pi)

        return means, disps,pis


def negative_binomial_loss(mean, disp, scale_factor, x):
    eps = 1e-12
    mean = mean * scale_factor

    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
        x * (torch.log(disp + eps) - torch.log(mean + eps))
    )
    nb_final = t1 + t2
    result = nb_final

    return result


def poisson_loss(mean, scale_factor, x):
    eps = 1e-12
    mean = mean * scale_factor

    poisson = mean - x * torch.log(mean + eps) + torch.lgamma(x + 1)

    return poisson

def zinb_loss(mean, disp, pi, scale_factor, x):
    eps = 1e-12
    mean = mean * scale_factor
    pi = torch.clamp(pi, eps, 1.0 - eps)
    nb_loss = negative_binomial_loss(mean, disp, 1.0, x)
    p_nb = torch.exp(-nb_loss.clamp(max=1e6))
    is_zero = (x <= 0.0)
    p_zero = pi + (1.0 - pi) * p_nb
    loss_zero = -torch.log(p_zero + eps)
    loss_nonzero = -torch.log(1.0 - pi + eps) + nb_loss
    loss = torch.where(is_zero, loss_zero, loss_nonzero)
    return loss

def reconstruction_loss(mean, disp, scale_factor, x, data_type, pi=None, reduction="mean"):
    if data_type in ["RNA", "ADT"]:
        loss = zinb_loss(mean, disp, pi, scale_factor, x)
    elif data_type == "ATAC":
        loss = poisson_loss(mean, scale_factor, x)
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    return loss
