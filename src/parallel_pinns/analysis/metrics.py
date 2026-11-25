"""Metricas de acuracia (E_L2, E_H1, L_PDE, espectrais), balanco global e time-to-solution."""
import numpy as np
import torch

from ..physics import geometry
from ..physics.operators import laplaciano
from .fourier import coeficientes_seno, metricas_espectrais


def avaliar(u_func, prob, N=63):
    # u_func: funcao (N,2) -> (N,1). Pode ser a rede ou qualquer aproximacao.
    xy, X, Y = geometry.grade_interior(N)
    xy = xy.clone().requires_grad_(True)
    x, y = xy[:, 0:1], xy[:, 1:2]

    u_ap = u_func(xy)
    u_ex = prob["u"](x, y)

    grad_ap = torch.autograd.grad(u_ap, xy, torch.ones_like(u_ap), create_graph=True)[0]
    grad_ex = torch.autograd.grad(u_ex, xy, torch.ones_like(u_ex), create_graph=True)[0]
    residuo = -laplaciano(u_ap, xy) - prob["f"](x, y)

    e, ge = (u_ap - u_ex).detach(), (grad_ap - grad_ex).detach()
    ue, gue = u_ex.detach(), grad_ex.detach()

    m = {
        "L2": (torch.sqrt((e**2).sum()) / torch.sqrt((ue**2).sum())).item(),
        "H1": (torch.sqrt((e**2).sum() + (ge**2).sum()) / torch.sqrt((ue**2).sum() + (gue**2).sum())).item(),
        "PDE": (residuo.detach()**2).mean().item(),
        "U": u_ap.detach().reshape(N, N).cpu().numpy().astype(np.float64),
    }
    if "n" in prob:
        a = coeficientes_seno(m["U"])
        m["E_n"], m["E_1"], m["E_esp"] = metricas_espectrais(a, prob["n"], prob["eps"])
        m["coef"] = a
    return m


def balanco_global(modelo, prob, Nq=64):
    # Teorema da divergencia: -lap u = f  =>  integral de f em Omega = - integral de du/dn na borda.
    # Devolve |int f + int du_theta/dn| / int |du*/dn|  (0 para a solucao exata).
    t, w = np.polynomial.legendre.leggauss(Nq)
    t = torch.tensor((t + 1) / 2, dtype=torch.float32).unsqueeze(1)
    w = torch.tensor(w / 2, dtype=torch.float32).unsqueeze(1)
    X, Y = torch.meshgrid(t.ravel(), t.ravel(), indexing="ij")
    integral_f = (prob["f"](X, Y) * (w @ w.T)).sum()

    z, o = torch.zeros_like(t), torch.ones_like(t)
    lados = [(torch.cat([z, t], 1), 0, -1.0), (torch.cat([o, t], 1), 0, 1.0),     # x = 0 e x = 1
             (torch.cat([t, z], 1), 1, -1.0), (torch.cat([t, o], 1), 1, 1.0)]     # y = 0 e y = 1
    fluxo_rede, fluxo_exato_abs = 0.0, 0.0
    for pts, comp, sinal in lados:
        pts = pts.clone().requires_grad_(True)
        for func, alvo in [(modelo, "rede"), (lambda p: prob["u"](p[:, 0:1], p[:, 1:2]), "exato")]:
            u = func(pts)
            dn = sinal * torch.autograd.grad(u, pts, torch.ones_like(u))[0][:, comp:comp + 1]
            if alvo == "rede":
                fluxo_rede += (dn * w).sum().item()
            else:
                fluxo_exato_abs += (dn.abs() * w).sum().item()
    return abs(integral_f.item() + fluxo_rede) / fluxo_exato_abs


CHAVES_METRICAS = ("L2", "H1", "PDE", "E_n", "E_1", "E_esp")


def metricas_finais(rede, prob, N_aval=63, invariantes=True):
    # Metricas gravadas no .json de cada execucao: acuracia, espectrais (P2) e, com invariantes=True, as
    # verificacoes de consistencia fisica: balanco global (todas) e saltos na interface (XPINN). Calculadas depois
    # do treino, nao mudam nada nele.
    from .interface import saltos_interface

    m = avaliar(rede, prob, N_aval)
    metricas = {k: float(v) for k, v in m.items() if k in CHAVES_METRICAS}
    if invariantes:
        metricas["E_bal"] = float(balanco_global(rede, prob))
        if hasattr(rede, "sub_redes"):
            metricas["J_u"], metricas["J_n"] = (float(v) for v in saltos_interface(rede, prob))
    return metricas


def historico_avaliacoes(hist):
    # Parte do historico usada nas figuras de convergencia e no time-to-solution
    return {k: hist[k] for k in ("aval_it", "aval_tempo", "L2", "E_n", "E_1") if k in hist}


def time_to_solution(hist, chave, limiar):
    # Tempo de relogio a partir do qual hist[chave] fica abaixo do limiar ATE O FIM do treino.
    # (So "a primeira vez abaixo" nao serve: E_n pode cruzar o limiar de passagem
    #  enquanto o coeficiente a_nn ainda esta oscilando.)
    valores = hist[chave]
    for k in range(len(valores)):
        if all(v < limiar for v in valores[k:]):
            return hist["aval_tempo"][k]
    return None
