"""Operadores diferenciais por diferenciacao automatica (exata, sem erro de truncamento)."""
import torch


def laplaciano(u, xy):
    # u: (N,1) saida da rede (ou da funcao exata), xy: (N,2) com requires_grad
    grad = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grad[:, 0:1]
    u_y = grad[:, 1:2]

    u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    return u_xx + u_yy
