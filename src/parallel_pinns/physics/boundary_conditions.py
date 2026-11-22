"""Contorno de Dirichlet imposto por construcao (sensibilidade da secao 9 do notebook).

O nucleo do TCC usa o contorno fraco (penalidade na perda), que fica em ``training.trainer``.
"""
import torch

from .geometry import distancia_contorno


class ContornoExato(torch.nn.Module):
    # u(x, y) = D(x, y) * NN(x, y) + G(x, y)
    # D zera a rede na borda e G carrega o valor de Dirichlet: u = g na borda por construcao
    def __init__(self, rede, G):
        super().__init__()
        self.rede = rede
        self.G = G

    def forward(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return distancia_contorno(x, y) * self.rede(xy) + self.G(x, y)


class ContornoTransfinito(torch.nn.Module):
    # u = NN - T[NN] + T[G], com T a interpolacao transfinita (Coons) dos valores na borda.
    # Na borda, T[F] = F, entao u = G = g por construcao. Nao ha divisao por D:
    # se NN = u*, entao u = u*, e a rede pode representar a propria solucao.
    def __init__(self, rede, G):
        super().__init__()
        self.rede = rede
        self.G = G

    @staticmethod
    def T(F, x, y):
        z, o = torch.zeros_like(x), torch.ones_like(x)
        cantos = F(torch.tensor([[0.0], [1.0], [0.0], [1.0]], device=x.device), torch.tensor([[0.0], [0.0], [1.0], [1.0]], device=x.device))
        c00, c10, c01, c11 = cantos[0], cantos[1], cantos[2], cantos[3]
        return ((1 - x) * F(z, y) + x * F(o, y) + (1 - y) * F(x, z) + y * F(x, o)
                - ((1 - x) * (1 - y) * c00 + x * (1 - y) * c10 + (1 - x) * y * c01 + x * y * c11))

    def forward(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        NN = lambda a, b: self.rede(torch.cat([a, b], 1))
        return NN(x, y) - self.T(NN, x, y) + self.T(self.G, x, y)
