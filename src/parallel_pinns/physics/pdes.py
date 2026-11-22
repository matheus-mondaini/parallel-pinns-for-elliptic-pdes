"""Benchmarks L0, P0, P1, P2(n) e L1(m): solucao exata u*, fonte f e levantamento de contorno G."""
import math

import torch


def phi(m, n, x, y):
    # Autofuncao do Laplaciano de Dirichlet no quadrado unitario
    return torch.sin(m * torch.pi * x) * torch.sin(n * torch.pi * y)


def L0_u(x, y):
    return torch.sin(torch.pi * x) * torch.sinh(torch.pi * y) / math.sinh(math.pi)


def L0_f(x, y):
    return torch.zeros_like(x)


def P0_u(x, y):
    # A x(1-x)y(1-y)(x+y): grau 3 em cada variavel. u*/D = A(x + y) deixa de ser constante.
    # A = 1/0.06912 leva o maximo (em x = y = 0.6) a 1, a mesma amplitude dos outros benchmarks.
    return x * (1 - x) * y * (1 - y) * (x + y) / 0.06912


def P0_f(x, y):
    # -lap(u) = A [2y(1-y)(3x + y - 1) + 2x(1-x)(x + 3y - 1)]
    return (2 * y * (1 - y) * (3 * x + y - 1) + 2 * x * (1 - x) * (x + 3 * y - 1)) / 0.06912


def P1_u(x, y):
    return phi(1, 1, x, y)


def P1_f(x, y):
    return 2 * torch.pi**2 * phi(1, 1, x, y)


def P2_u(x, y, n, eps=0.1):
    return phi(1, 1, x, y) + eps * phi(n, n, x, y)


def P2_f(x, y, n, eps=0.1):
    return 2 * torch.pi**2 * phi(1, 1, x, y) + 2 * eps * n**2 * torch.pi**2 * phi(n, n, x, y)


def L1_u(x, y, m):
    return torch.sin(m * torch.pi * x) * torch.sinh(m * torch.pi * y) / math.sinh(m * math.pi)


def L1_f(x, y, m):
    return torch.zeros_like(x)


def criar_problema(nome, n=None, eps=0.1, m=None):
    # Junta u*, f, o levantamento de contorno G e os parametros em um dicionario simples.
    # G e qualquer funcao suave com G = g na borda (usada no contorno exato u = D * NN + G).
    zero = lambda x, y: torch.zeros_like(x)
    if nome == "L0":
        return {"nome": "L0", "u": L0_u, "f": L0_f,
                "G": lambda x, y: y * torch.sin(torch.pi * x)}          # 0 em x=0, x=1, y=0; sin(pi x) em y=1
    if nome == "P0":
        return {"nome": "P0", "u": P0_u, "f": P0_f, "G": zero}
    if nome == "P1":
        return {"nome": "P1", "u": P1_u, "f": P1_f, "G": zero}
    if nome == "P2":
        return {"nome": f"P2(n={n})", "n": n, "eps": eps,
                "u": lambda x, y: P2_u(x, y, n, eps),
                "f": lambda x, y: P2_f(x, y, n, eps), "G": zero}
    if nome == "L1":
        return {"nome": f"L1(m={m})",
                "u": lambda x, y: L1_u(x, y, m),
                "f": lambda x, y: L1_f(x, y, m),
                "G": lambda x, y: y * torch.sin(m * torch.pi * x)}
    raise ValueError(f"benchmark desconhecido: {nome!r} (use L0, P0, P1, P2 ou L1)")
