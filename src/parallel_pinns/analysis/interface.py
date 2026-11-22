"""Diagnosticos do XPINN: saltos na interface e residuo por subdominio."""
import numpy as np
import torch

from ..models.xpinn import pontos_interface
from ..physics.operators import laplaciano


def saltos_interface(xpinn, prob, N_I=200):
    # Saltos de u e de du/dn nas interfaces, medidos na avaliacao (nao na perda).
    # Normalizados pelo valor RMS de u* e de |grad u*| nos mesmos pontos.
    xy_I, ka, kb, normal = pontos_interface(xpinn.M, N_I)
    nseg = xy_I.shape[0]
    xy = xy_I.reshape(-1, 2).clone().requires_grad_(True)
    vista = xy.view(nseg, N_I, 2)
    u_a = xpinn.sub_redes(vista, ka).reshape(-1, 1)
    u_b = xpinn.sub_redes(vista, kb).reshape(-1, 1)
    g_a = torch.autograd.grad(u_a, xy, torch.ones_like(u_a), retain_graph=True)[0]
    g_b = torch.autograd.grad(u_b, xy, torch.ones_like(u_b))[0]
    n_pt = normal.repeat_interleave(N_I).unsqueeze(1)
    u_ex = prob["u"](xy[:, 0:1], xy[:, 1:2])
    g_ex = torch.autograd.grad(u_ex, xy, torch.ones_like(u_ex))[0]
    rms = lambda v: torch.sqrt(torch.mean(v**2)).item()
    salto_u = rms(u_a - u_b) / rms(u_ex)
    salto_fluxo = rms(g_a.gather(1, n_pt) - g_b.gather(1, n_pt)) / rms(g_ex.norm(dim=1))
    return salto_u, salto_fluxo


def residuo_por_subdominio(modelo, prob, M, Ng=128):
    # Residuo r = -lap(u) - f numa grade de centros de celula Ng x Ng (nenhum ponto cai numa interface).
    # Para cada subdominio k de uma particao M x M: L_PDE_k = media de r^2 e I_k = integral de r em Omega_k,
    # relativa a int |du*/dn| na borda (a mesma escala do E_bal).
    # Teorema da divergencia: soma_k I_k = [-oint du/dn - int f - (saltos de du/dn nas interfaces)] / escala,
    # entao E_bal grande com saltos pequenos exige |soma_k I_k| grande.
    c = (torch.arange(Ng) + 0.5) / Ng
    X, Y = torch.meshgrid(c, c, indexing="ij")
    xy = torch.stack([X.ravel(), Y.ravel()], 1).requires_grad_(True)
    u = modelo(xy)
    r = (-laplaciano(u, xy) - prob["f"](xy[:, 0:1], xy[:, 1:2])).detach().ravel()
    k = (torch.clamp((xy[:, 0] * M).floor().long(), 0, M - 1) * M
         + torch.clamp((xy[:, 1] * M).floor().long(), 0, M - 1)).detach()

    # Escala: int |du*/dn| na borda (Gauss-Legendre, como em balanco_global)
    t, w = np.polynomial.legendre.leggauss(64)
    t = torch.tensor((t + 1) / 2, dtype=torch.float32).unsqueeze(1)
    w = torch.tensor(w / 2, dtype=torch.float32).unsqueeze(1)
    z, o = torch.zeros_like(t), torch.ones_like(t)
    escala = 0.0
    for pts, comp in [(torch.cat([z, t], 1), 0), (torch.cat([o, t], 1), 0), (torch.cat([t, z], 1), 1), (torch.cat([t, o], 1), 1)]:
        pts = pts.clone().requires_grad_(True)
        ue = prob["u"](pts[:, 0:1], pts[:, 1:2])
        escala += (torch.autograd.grad(ue, pts, torch.ones_like(ue))[0][:, comp:comp + 1].abs() * w).sum().item()

    L_k = torch.zeros(M * M).index_add_(0, k, r**2) / torch.bincount(k, minlength=M * M)
    I_k = torch.zeros(M * M).index_add_(0, k, r) / Ng**2 / escala
    # Matrizes M x M com y crescendo para cima (linha 0 = faixa de cima)
    em_grade = lambda v: v.reshape(M, M).T.flip(0).cpu().numpy()
    return em_grade(L_k), em_grade(I_k)
