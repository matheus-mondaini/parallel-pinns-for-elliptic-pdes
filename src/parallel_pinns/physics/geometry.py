"""Dominio (0,1)^2: pontos de colocacao (LHS), pontos de contorno, grade de avaliacao e funcao distancia."""
import torch
from scipy.stats import qmc


def grade_interior(N=63):
    x = torch.arange(1, N + 1, dtype=torch.float32) / (N + 1)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    xy = torch.stack([X.ravel(), Y.ravel()], dim=1)
    return xy, X, Y


def pontos_colocacao(N_f, semente):
    lhs = qmc.LatinHypercube(d=2, seed=semente)
    return torch.tensor(lhs.random(N_f), dtype=torch.float32)


def pontos_contorno(N_b):
    # N_b/4 pontos por lado, sem repetir os cantos
    s = torch.linspace(0, 1, N_b // 4 + 1)[:-1]
    zero, um = torch.zeros_like(s), torch.ones_like(s)
    lados = [torch.stack([s, zero], 1),        # y = 0
             torch.stack([um, s], 1),          # x = 1
             torch.stack([1 - s, um], 1),      # y = 1
             torch.stack([zero, 1 - s], 1)]    # x = 0
    return torch.cat(lados, 0)


def distancia_contorno(x, y):
    # D(x, y) = x(1-x)y(1-y): zero nos quatro lados, positiva no interior
    return x * (1 - x) * y * (1 - y)
