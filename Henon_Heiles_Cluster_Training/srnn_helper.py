from __future__ import annotations
import torch
import torch.nn as nn
from torch.autograd import grad

from helper import MLP, IntegratorBase


class SRNN_Network(nn.Module):
    """Fixed-lambda counterpart of Hamiltonian_MLP_Network: V_theta2(q) with the
    lambda channel removed. K_theta1(p) is unchanged."""

    def __init__(self, kin_hidden_dim: int, kin_n_hidden: int, pot_hidden_dim: int, pot_n_hidden: int,
                 device: torch.device | None = None):
        super().__init__()
        self.device = device
        self.V_net = MLP(in_dim=2, out_dim=1, hidden_dim=pot_hidden_dim, n_hidden=pot_n_hidden).to(device)
        self.K_net = MLP(in_dim=2, out_dim=1, hidden_dim=kin_hidden_dim, n_hidden=kin_n_hidden).to(device)

    def forward(self, p, q, params=None):
        p, q = p.to(self.device), q.to(self.device)
        K = self.K_net(p)
        V = self.V_net(q)
        return K, V


class SRNNVerletIntegrator(IntegratorBase):
    """Same Verlet recurrence as VerletIntegrator, but V_net takes q only (no params)."""

    def step(self, p: torch.Tensor, q: torch.Tensor, params=None):
        q, p = q.to(self.device), p.to(self.device)
        q.requires_grad_(True)
        p.requires_grad_(True)
        V = self.model.V_net(q)
        dpdt = -grad(V.sum(), q, create_graph=True)[0]
        p_half = p + dpdt * (self.dt / 2)
        K = self.model.K_net(p_half)
        dqdt = grad(K.sum(), p_half, create_graph=True)[0]
        q_next = q + dqdt * self.dt
        q_next.requires_grad_(True)
        V2 = self.model.V_net(q_next)
        dpdt2 = -grad(V2.sum(), q_next, create_graph=True)[0]
        p_next = p_half + dpdt2 * (self.dt / 2)
        return p_next, q_next


__all__ = ["SRNN_Network", "SRNNVerletIntegrator"]
