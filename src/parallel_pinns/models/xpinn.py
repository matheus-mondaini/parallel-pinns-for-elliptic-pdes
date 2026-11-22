"""XPINN: M x M sub-redes, uma por subdominio, com os pesos empilhados para avaliar varias de uma vez."""
import math

import torch

from ..physics.geometry import distancia_contorno


class XPINN(torch.nn.Module):
    # M x M sub-redes, uma por subdominio. Os pesos ficam empilhados em tensores (S, entrada, saida)
    # para calcular varias sub-redes de uma vez com torch.bmm.
    # Com G definido, o contorno e exato: u = D(x, y) * NN_k(x, y) + G(x, y), com D e G GLOBAIS.
    def __init__(self, M, largura, camadas_ocultas=4, G=None):
        super().__init__()
        self.M, self.S, self.G = M, M * M, G
        tamanhos = [2] + [largura] * camadas_ocultas + [1]
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for a, c in zip(tamanhos[:-1], tamanhos[1:]):
            desvio = math.sqrt(2.0 / (a + c))                      # Glorot/Xavier normal, igual a monolitica
            self.W.append(torch.nn.Parameter(torch.randn(self.S, a, c) * desvio))
            self.b.append(torch.nn.Parameter(torch.zeros(self.S, 1, c)))
        # Canto inferior esquerdo de cada subdominio k = i*M + j  (i: coluna em x, j: linha em y)
        i, j = torch.meshgrid(torch.arange(M), torch.arange(M), indexing="ij")
        self.register_buffer("canto", torch.stack([i.ravel(), j.ravel()], 1).float().unsqueeze(1) / M)

    def subdominio(self, xy):
        # Indice do subdominio de cada ponto (pontos exatamente na interface vao para o de cima/direita)
        i = torch.clamp((xy[:, 0] * self.M).floor().long(), 0, self.M - 1)
        j = torch.clamp((xy[:, 1] * self.M).floor().long(), 0, self.M - 1)
        return i * self.M + j

    def sub_redes(self, xy, ks):
        # xy: (B, P, 2) em coordenadas GLOBAIS; ks: (B,) qual sub-rede avalia cada lote
        # Renormalizacao para coordenadas locais [-1, 1]^2 (secao 12.2 do documento)
        h = 2 * self.M * (xy - self.canto[ks]) - 1
        for camada in range(len(self.W)):
            h = torch.bmm(h, self.W[camada][ks]) + self.b[camada][ks]
            if camada < len(self.W) - 1:
                h = torch.tanh(h)
        if self.G is not None:
            x, y = xy[..., 0:1], xy[..., 1:2]
            h = distancia_contorno(x, y) * h + self.G(x, y)
        return h                                                    # (B, P, 1)

    def forward(self, xy):
        # Avaliacao em pontos quaisquer (N, 2): cada ponto e calculado so pela sub-rede do seu subdominio
        k = self.subdominio(xy)
        partes, ordem = [], []
        for s in range(self.S):
            idx = torch.nonzero(k == s).squeeze(1)
            if len(idx) > 0:
                partes.append(self.sub_redes(xy[idx].unsqueeze(0), torch.tensor([s]))[0])
                ordem.append(idx)
        ordem = torch.cat(ordem)
        inversa = torch.empty_like(ordem)
        inversa[ordem] = torch.arange(len(ordem))
        return torch.cat(partes)[inversa]


def agrupar_por_subdominio(xy, k, S):
    # Separa os pontos por subdominio num tensor (S, P_max, 2) com mascara (os lotes tem tamanhos diferentes)
    contagem = torch.bincount(k, minlength=S)
    P = int(contagem.max())
    lote = torch.zeros(S, P, 2)
    mascara = torch.zeros(S, P, 1)
    for s in range(S):
        pts = xy[k == s]
        lote[s, :len(pts)] = pts
        lote[s, len(pts):] = pts[0] if len(pts) > 0 else 0.5    # preenchimento (ignorado pela mascara)
        mascara[s, :len(pts)] = 1.0
    return lote, mascara


def pontos_interface(M, N_I):
    # N_I pontos em cada segmento de interface. Devolve pontos (nseg, N_I, 2), o par de sub-redes (a, b)
    # e a direcao normal ao segmento (0 = x, 1 = y)
    t = (torch.arange(N_I) + 0.5) / N_I
    pts, ka, kb, normal = [], [], [], []
    for i in range(1, M):                     # interfaces verticais x = i/M
        for j in range(M):
            y = (j + t) / M
            pts.append(torch.stack([torch.full_like(y, i / M), y], 1))
            ka.append((i - 1) * M + j)
            kb.append(i * M + j)
            normal.append(0)                  # normal na direcao x
    for j in range(1, M):                     # interfaces horizontais y = j/M
        for i in range(M):
            x = (i + t) / M
            pts.append(torch.stack([x, torch.full_like(x, j / M)], 1))
            ka.append(i * M + (j - 1))
            kb.append(i * M + j)
            normal.append(1)                  # normal na direcao y
    if not pts:                               # M = 1: sem interfaces
        vazio = torch.zeros(0, dtype=torch.long)
        return torch.zeros(0, N_I, 2), vazio, vazio, vazio
    return torch.stack(pts), torch.tensor(ka), torch.tensor(kb), torch.tensor(normal)
