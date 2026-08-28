

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from sklearn.model_selection import train_test_split as sk_train_test_split
from typing import Optional

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.use_deterministic_algorithms(False)


def F(p: torch.Tensor, q: torch.Tensor, alpha_1: float, alpha_2: float) -> torch.Tensor:
    p1 = p[:, 0:1]; p2 = p[:, 1:]
    q1 = q[:, 0:1]; q2 = q[:, 1:]
    dp1dt = -q1 - 2 * alpha_1 * q1 * q2
    dp2dt = -q2 - alpha_1 * q1**2 + alpha_2 * q2**2
    dq1dt = p1
    dq2dt = p2
    return torch.cat((dp1dt, dp2dt, dq1dt, dq2dt), dim=1)

def generate_initial_conditions(alpha1: float, alpha2: float, in_conds: int) -> torch.Tensor:
    dat_ = torch.empty((in_conds, 4), dtype=torch.float32)
    i = 0
    while i < in_conds:
        p0 = 2 * torch.rand(2) - 1
        q0 = 2 * torch.rand(2) - 1
        E = 0.5 * (p0 @ p0 + q0 @ q0) + alpha1 * (q0[0]**2) * q0[1] - alpha2 * (q0[1]**3) / 3
        if E <= 1/6:
            dat_[i, :2] = p0
            dat_[i, 2:] = q0
            i += 1
    return dat_

def _like(t: torch.Tensor):
    return dict(device=t.device, dtype=t.dtype)

def generate_trajectories(p0: torch.Tensor, q0: torch.Tensor, Ffun, alpha1: float, alpha2: float,
                          T: int, dt: float, coarsening_factor: int = 1) -> torch.Tensor:
    fine = torch.empty((T*coarsening_factor, p0.shape[0], 2*p0.shape[1]), **_like(p0))
    dtau = dt / coarsening_factor
    p, q = p0, q0
    dim = p0.shape[1]
    time_drvt = Ffun(p, q, alpha1, alpha2); dpdt = time_drvt[:, :dim]
    for i in range(T * coarsening_factor):
        p_half = p + dpdt * (dtau / 2)
        fine[i, :, :dim] = p
        fine[i, :, dim:] = q
        time_drvt = Ffun(p_half, q, alpha1, alpha2); dqdt = time_drvt[:, dim:]
        q_next = q + dqdt * dtau
        time_drvt = Ffun(p_half, q_next, alpha1, alpha2); dpdt = time_drvt[:, :dim]
        p_next = p_half + dpdt * (dtau / 2)
        p, q = p_next, q_next
    traj = fine[torch.arange(T, device=p0.device) * coarsening_factor, :, :]
    return traj  

def split_trajectory(trajectory: torch.Tensor, T_window: int, N: int):
    L = len(trajectory)
    splits = L - T_window + 1
    B, D = trajectory.shape[1], trajectory.shape[2]
    out = torch.empty((splits, N, B, D), **_like(trajectory))
    idx = torch.empty((splits, N), dtype=torch.long, device=trajectory.device)
    for i in range(splits):
        idx[i, 0] = 0
        if N > 1:
            rand = torch.randperm(T_window - 1, device=trajectory.device)[:N-1] + 1
            idx[i, 1:] = rand
        idx[i] = idx[i].sort()[0]
        window = trajectory[i:i+T_window]          
        out[i] = window[idx[i]]                   
    return out, idx

def compute_within_variance_torch(trajectories: torch.Tensor, burn_in: int = 0) -> torch.Tensor:
    T, B, D = trajectories.shape
    Tp = T - burn_in
    data = trajectories[burn_in:, :, :]
    means = data.mean(dim=0, keepdim=True)
    centered = data - means
    num = centered.pow(2).sum(dim=(0, 1))
    den = B * (Tp - 1)
    return num / den  

def generate_ou_noise_exact(T: int, B: int, var_eta_per_channel: torch.Tensor,
                            a: float, *, device=None, dtype=None,
                            generator: Optional[torch.Generator] = None) -> torch.Tensor:
    D = var_eta_per_channel.numel()
    device = device or var_eta_per_channel.device
    dtype = dtype or var_eta_per_channel.dtype
    ou = torch.empty((T, B, D), device=device, dtype=dtype)
    std0 = var_eta_per_channel.clamp_min(0).sqrt()
    innov_std = (var_eta_per_channel * (1.0 - a*a)).clamp_min(0).sqrt()
    eta = torch.randn((B, D), device=device, dtype=dtype, generator=generator) * std0
    ou[0] = eta
    for t in range(1, T):
        xi = torch.randn((B, D), device=device, dtype=dtype, generator=generator)
        eta = a * eta + innov_std * xi
        ou[t] = eta
    return ou

def generate_data(alphas: torch.Tensor, Ffun, L: int, T_window: int, N: int, *,
                  dt: float = 0.1, in_conds: int = 8, coarsening_factor: int = 1,
                  nsr: float = 0.10, theta: float = 0.5, burn_in_var: int = 0,
                  apply_ou_noise: bool = True, device: Optional[torch.device] = None,
                  dtype: torch.dtype = torch.float32,
                  seed_ou: Optional[int] = None):
   
    if device is None:
        device = alphas.device
    gen_ou = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    if seed_ou is not None: gen_ou.manual_seed(seed_ou)

    K = alphas.shape[0]
    clean_list = []
    for k in range(K):
        a1 = float(alphas[k, 0].item()); a2 = float(alphas[k, 1].item())
        ic = generate_initial_conditions(a1, a2, in_conds)                 
        p0 = ic[:, :2].to(device=device, dtype=dtype)
        q0 = ic[:, 2:].to(device=device, dtype=dtype)
        traj_clean = generate_trajectories(p0, q0, Ffun, a1, a2, L, dt, coarsening_factor) 
        clean_list.append(traj_clean)

    all_clean = torch.cat(clean_list, dim=1)  
    var_within_all = compute_within_variance_torch(all_clean, burn_in=burn_in_var)  
    var_eta = nsr * var_within_all.to(device=device, dtype=dtype)
    a = float(torch.exp(torch.tensor(-theta * dt)).item())

    traj_batches_all, sampled_idx_all, params_all = [], [], []
    for k in range(K):
        traj = clean_list[k].clone()
        Lc, Bk, D = traj.shape
        if apply_ou_noise:
            ou = generate_ou_noise_exact(Lc, Bk, var_eta, a, device=traj.device, dtype=traj.dtype, generator=gen_ou)
            traj = traj + ou
        split_traj, sampled_idx = split_trajectory(traj, T_window, N)  
        splits = split_traj.shape[0]
        tb = split_traj.permute(1, 0, 2, 3).contiguous().view(N, splits * Bk, D)  
        traj_batches_all.append(tb)
        idx_rep = sampled_idx.repeat_interleave(Bk, dim=0)  
        sampled_idx_all.append(idx_rep)
        a1, a2 = alphas[k, 0].item(), alphas[k, 1].item()
        params_k = torch.tensor([[a1, a2]], device=traj.device, dtype=traj.dtype).repeat(splits * Bk, 1)
        params_all.append(params_k)

    trajectories = torch.cat(traj_batches_all, dim=1)     
    params       = torch.cat(params_all,        dim=0)      
    indices      = torch.cat(sampled_idx_all,   dim=0)     

    n = trajectories.size(1)
    perm = torch.randperm(n, device=trajectories.device)
    trajectories = trajectories[:, perm, :]
    params       = params[perm, :]
    indices      = indices[perm, :]

    return trajectories, params, indices

def train_test_split(trajectories: torch.Tensor, params: torch.Tensor, indices: torch.Tensor, val_size: float = 0.25):
    N = trajectories.shape[1]
    all_idx = torch.arange(N)
    tr_idx, va_idx = sk_train_test_split(all_idx, test_size=val_size, random_state=42)
    train_trajectories = trajectories[:, tr_idx, :]
    val_trajectories   = trajectories[:, va_idx, :]
    train_params = params[tr_idx, :]
    val_params   = params[va_idx, :]
    train_indices = indices[tr_idx, :]
    val_indices   = indices[va_idx, :]
    return (train_trajectories, train_params, train_indices), (val_trajectories, val_params, val_indices)


class IntegratorBase(nn.Module):
    def __init__(self, model: nn.Module, dt: float):
        super().__init__()
        self.model = model
        self.dt = dt
        self.device = next(model.parameters()).device

class VerletIntegrator(IntegratorBase):
    def step(self, p: torch.Tensor, q: torch.Tensor, params: torch.Tensor):
        q, p, params = q.to(self.device), p.to(self.device), params.to(self.device)
        q.requires_grad_(True)
        p.requires_grad_(True)
        V = self.model.V_net(torch.cat((q, params), dim=1))
        dpdt = -grad(V.sum(), q, create_graph=True)[0]
        p_half = p + dpdt * (self.dt / 2)
        K = self.model.K_net(p_half)
        dqdt = grad(K.sum(), p_half, create_graph=True)[0]
        q_next = q + dqdt * self.dt
        q_next.requires_grad_(True)
        V2 = self.model.V_net(torch.cat((q_next, params), dim=1))
        dpdt2 = -grad(V2.sum(), q_next, create_graph=True)[0]
        p_next = p_half + dpdt2 * (self.dt / 2)
        return p_next, q_next

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 50, n_hidden: int = 1):
        super().__init__()
        act = nn.Tanh()
        layers = [nn.Linear(in_dim, hidden_dim), act]
        for _ in range(1, n_hidden):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Hamiltonian_MLP_Network(nn.Module):
    def __init__(self, kin_hidden_dim: int, kin_n_hidden: int, pot_hidden_dim: int, pot_n_hidden: int, device: torch.device | None = None):
        super().__init__()
        self.device = device
        self.V_net = MLP(in_dim=4, out_dim=1, hidden_dim=pot_hidden_dim, n_hidden=pot_n_hidden).to(device)
        self.K_net = MLP(in_dim=2, out_dim=1, hidden_dim=kin_hidden_dim, n_hidden=kin_n_hidden).to(device)
    def forward(self, p, q, params):
        p, q, params = p.to(self.device), q.to(self.device), params.to(self.device)
        K = self.K_net(p)
        V = self.V_net(torch.cat((q, params), dim=1))
        return K, V

HamiltonianMLPNetwork = Hamiltonian_MLP_Network

def residuals(train_trajectories: torch.Tensor, train_params: torch.Tensor, train_instants: torch.Tensor,
              T: int, integrator: VerletIntegrator) -> torch.Tensor:
    
    batch_size = train_trajectories.shape[1]
    N_idx = train_instants.shape[1]
    p_pred = train_trajectories[0, :, 0:2].clone().to(integrator.device).requires_grad_(True)
    q_pred = train_trajectories[0, :, 2:4].clone().to(integrator.device).requires_grad_(True)
    params = train_params.to(integrator.device)

    p_preds = [p_pred]; q_preds = [q_pred]
    for _ in range(1, T):
        p_pred, q_pred = integrator.step(p_pred, q_pred, params)
        p_preds.append(p_pred); q_preds.append(q_pred)

    p_preds = torch.stack(p_preds)    
    q_preds = torch.stack(q_preds)    

    idx_b = torch.arange(batch_size, device=integrator.device)
    idx_t = train_instants.to(device=integrator.device, dtype=torch.long)

    sampled_p = torch.stack([p_preds[idx_t[:, i], idx_b] for i in range(N_idx)], dim=0)
    sampled_q = torch.stack([q_preds[idx_t[:, i], idx_b] for i in range(N_idx)], dim=0)

    p_true = train_trajectories[:, :, 0:2].to(integrator.device)
    q_true = train_trajectories[:, :, 2:4].to(integrator.device)

    loss_p = torch.mean((sampled_p - p_true) ** 2)
    loss_q = torch.mean((sampled_q - q_true) ** 2)
    return loss_p + loss_q


def F_batch(pq: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    p1 = pq[:, 0:1]; p2 = pq[:, 1:2]
    q1 = pq[:, 2:3]; q2 = pq[:, 3:4]
    alpha_1 = params[:, 0:1]; alpha_2 = params[:, 1:2]
    dp1dt = -q1 - 2 * alpha_1 * q1 * q2
    dp2dt = -q2 - alpha_1 * q1**2 + alpha_2 * q2**2
    dq1dt = p1
    dq2dt = p2
    return torch.cat((dp1dt, dp2dt, dq1dt, dq2dt), dim=1)

def generate_trajectories_batch(in_conds: torch.Tensor, params: torch.Tensor,
                                T: int, dt: float, coarsening_factor: int = 1) -> torch.Tensor:
   
    n_i = in_conds.shape[0]
    dim = in_conds.shape[1] // 2
    fine = torch.empty((T * coarsening_factor, n_i, dim * 2), device=in_conds.device, dtype=in_conds.dtype)
    dtau = dt / coarsening_factor

    p = in_conds[:, :dim]
    q = in_conds[:, dim:]

    pq = torch.cat([p, q], dim=1)
    time_drvt = F_batch(pq, params)
    dpdt = time_drvt[:, :dim]

    for i in range(T * coarsening_factor):
        p_half = p + dpdt * (dtau / 2)
        fine[i, :, :dim] = p
        fine[i, :, dim:] = q

        pq_half = torch.cat([p_half, q], dim=1)
        time_drvt = F_batch(pq_half, params)
        dqdt = time_drvt[:, dim:]
        q_next = q + dqdt * dtau

        pq_half_next = torch.cat([p_half, q_next], dim=1)
        time_drvt = F_batch(pq_half_next, params)
        dpdt = time_drvt[:, :dim]
        p_next = p_half + dpdt * (dtau / 2)

        p, q = p_next, q_next

    trajectories = fine[torch.arange(T, device=in_conds.device) * coarsening_factor, :, :]
    return trajectories

def transform_trajectories_to_numpy(trajectories: torch.Tensor) -> np.ndarray:
    X = trajectories.permute(1, 0, 2)         
    X = torch.cat((X[:, :, 2:4], X[:, :, 0:2]), dim=2)  
    X = X.permute(2, 0, 1)                    
    return X.detach().cpu().numpy()

def E_time_series(X: np.ndarray, alphas: np.ndarray, betas: np.ndarray):
    m = 1.0
    kinetic = 0.5 * (X[2]**2 / m + X[3]**2 / m) 
    potential = 0.5 * (X[0]**2 + X[1]**2) + alphas[:, None] * (X[0]**2 * X[1]) - betas[:, None] * (X[1]**3) / 3.0
    energy = potential + kinetic
    return potential, kinetic, energy

def split_params(params_tensor: torch.Tensor):
    alpha, beta = params_tensor[:, 0], params_tensor[:, 1]
    return alpha.detach().cpu().numpy(), beta.detach().cpu().numpy()

def predict_trajectories(model: nn.Module, integrator: VerletIntegrator,
                         in_conds: torch.Tensor, params: torch.Tensor, T: int):
    p = in_conds[:, :2].to(model.device)
    q = in_conds[:, 2:].to(model.device)
    params = params.to(model.device)

    p_list, q_list, K_list, V_list = [], [], [], []
    for _ in range(T):
        K, V = model(p, q, params)
        p_list.append(p); q_list.append(q)
        K_list.append(K); V_list.append(V)
        p, q = integrator.step(p, q, params)

    p_preds = torch.stack(p_list, dim=0)
    q_preds = torch.stack(q_list, dim=0)
    K_preds = torch.stack(K_list, dim=0)
    V_preds = torch.stack(V_list, dim=0)
    return p_preds, q_preds, K_preds, V_preds



__all__ = [
    "F",
    "generate_data",
    "train_test_split",
    "Hamiltonian_MLP_Network",
    "HamiltonianMLPNetwork",   
    "VerletIntegrator",
    "residuals",
    "generate_trajectories_batch",
    "transform_trajectories_to_numpy",
    "E_time_series",
    "split_params",
    "predict_trajectories",
    "device",
]
