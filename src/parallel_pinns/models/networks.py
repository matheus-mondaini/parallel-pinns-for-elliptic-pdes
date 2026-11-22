"""MLP com tanh (inicializacao Glorot normal) e contagem de parametros para o XPINN iso-parametro."""
import torch


def criar_rede(largura=64, camadas_ocultas=4):
    camadas = [torch.nn.Linear(2, largura), torch.nn.Tanh()]
    for _ in range(camadas_ocultas - 1):
        camadas += [torch.nn.Linear(largura, largura), torch.nn.Tanh()]
    camadas += [torch.nn.Linear(largura, 1)]
    rede = torch.nn.Sequential(*camadas)
    for c in rede:
        if isinstance(c, torch.nn.Linear):
            torch.nn.init.xavier_normal_(c.weight)
            torch.nn.init.zeros_(c.bias)
    return rede


def n_parametros(largura, camadas_ocultas=4):
    # entrada 2 -> largura, (camadas_ocultas - 1) camadas largura -> largura, largura -> 1
    return (2 * largura + largura) + (camadas_ocultas - 1) * (largura**2 + largura) + (largura + 1)


def largura_iso_parametro(M, largura_mono=64, camadas_ocultas=4):
    # Maior largura tal que M^2 sub-redes somem no maximo os parametros da rede monolitica
    alvo = n_parametros(largura_mono, camadas_ocultas) / M**2
    w = 1
    while n_parametros(w + 1, camadas_ocultas) <= alvo:
        w += 1
    return w
