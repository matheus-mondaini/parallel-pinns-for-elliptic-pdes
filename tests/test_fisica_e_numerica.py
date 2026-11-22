"""Testes de unidade: benchmarks, operadores, geometria, metricas espectrais, FDM e lei de Amdahl.

Sao as verificacoes que o notebook fazia em celulas soltas (secoes 1 a 3, 11 e 10), agora automaticas.
"""
import math

import numpy as np
import pytest
import torch

from parallel_pinns.analysis import avaliar, balanco_global, coeficientes_seno, metricas_espectrais
from parallel_pinns.models import XPINN, largura_iso_parametro, n_parametros, pontos_interface
from parallel_pinns.physics import (ContornoExato, ContornoTransfinito, criar_problema, grade_interior, laplaciano,
                                    pontos_colocacao, pontos_contorno)
from parallel_pinns.references import resolver_fdm
from parallel_pinns.utils import ajustar_amdahl, amdahl, modo_float64

BENCHMARKS = [("L0", {}), ("P0", {}), ("P1", {}), ("P2", {"n": 5}), ("P2", {"n": 13}), ("L1", {"m": 3})]


@pytest.mark.parametrize("nome,kw", BENCHMARKS)
def test_fonte_e_menos_laplaciano_da_solucao(nome, kw):
    # -lap(u*) = f, conferido por diferenciacao automatica em float64 (MMS)
    prob = criar_problema(nome, **kw)
    xy = torch.rand(200, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).requires_grad_(True)
    u = prob["u"](xy[:, 0:1], xy[:, 1:2])
    f = prob["f"](xy[:, 0:1], xy[:, 1:2])
    assert torch.allclose(-laplaciano(u, xy), f, rtol=1e-9, atol=1e-8 * (1 + f.abs().max().item()))


@pytest.mark.parametrize("nome,kw", BENCHMARKS)
def test_levantamento_G_coincide_com_u_na_borda(nome, kw):
    prob = criar_problema(nome, **kw)
    xy = pontos_contorno(400).double()
    x, y = xy[:, 0:1], xy[:, 1:2]
    assert torch.allclose(prob["G"](x, y), prob["u"](x, y), atol=1e-12)


def test_benchmark_desconhecido():
    with pytest.raises(ValueError):
        criar_problema("P9")


def test_pontos():
    xy = pontos_colocacao(2000, semente=0)
    assert xy.shape == (2000, 2) and xy.min() >= 0 and xy.max() <= 1
    assert torch.equal(xy, pontos_colocacao(2000, semente=0))           # LHS reprodutivel pela semente
    b = pontos_contorno(400)
    assert b.shape == (400, 2) and len(torch.unique(b, dim=0)) == 400   # sem cantos repetidos
    assert torch.all(torch.any((b == 0) | (b == 1), dim=1))             # todos sobre algum lado


@pytest.mark.parametrize("Contorno", [ContornoExato, ContornoTransfinito])
def test_contorno_por_construcao(Contorno):
    prob = criar_problema("L0")
    torch.manual_seed(0)
    rede = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    u = Contorno(rede, prob["G"])
    b = pontos_contorno(200)
    assert torch.allclose(u(b), prob["u"](b[:, 0:1], b[:, 1:2]), atol=1e-6)


def test_metricas_espectrais_da_solucao_exata():
    # A DST-I recupera a_11 = 1 e a_nn = eps da solucao exata de P2(n): E_n, E_1 e E_esp no arredondamento
    prob = criar_problema("P2", n=11)
    xy, X, Y = grade_interior(63)
    U = prob["u"](X.double(), Y.double()).numpy()
    a = coeficientes_seno(U)
    E_n, E_1, E_esp = metricas_espectrais(a, 11, 0.1)
    assert max(E_n, E_1, E_esp) < 1e-12


def test_avaliar_e_balanco_da_solucao_exata():
    prob = criar_problema("P2", n=5)
    exata = lambda p: prob["u"](p[:, 0:1], p[:, 1:2])
    m = avaliar(exata, prob)
    assert m["L2"] == 0 and m["H1"] == 0 and m["E_n"] < 1e-5
    assert balanco_global(exata, prob) < 1e-4


@pytest.mark.parametrize("nome,kw", [("L0", {}), ("P1", {}), ("P2", {"n": 5})])
def test_fdm_converge_com_h2(nome, kw):
    prob = criar_problema(nome, **kw)
    erros = []
    for N in (15, 31, 63):
        X, Y, U = resolver_fdm(prob, N)
        U_ex = prob["u"](torch.from_numpy(X), torch.from_numpy(Y)).numpy()
        erros.append(np.linalg.norm(U - U_ex) / np.linalg.norm(U_ex))
    razoes = [erros[k] / erros[k + 1] for k in range(2)]
    assert all(3.5 < r < 4.5 for r in razoes)


def test_largura_iso_parametro_secao_11():
    # Valores impressos no notebook: monolitica 12737; M = 2 -> largura 31 (12404); M = 4 -> 15 (12496)
    assert n_parametros(64) == 12737
    assert [largura_iso_parametro(M) for M in (2, 4)] == [31, 15]
    for M, total in [(2, 12404), (4, 12496)]:
        rede = XPINN(M, largura_iso_parametro(M))
        assert sum(p.numel() for p in rede.parameters()) == total


def test_xpinn_avalia_cada_ponto_pela_sua_subrede():
    torch.manual_seed(0)
    rede = XPINN(2, 8)
    xy = torch.tensor([[0.1, 0.1], [0.9, 0.2], [0.3, 0.8], [0.7, 0.6]])
    k = rede.subdominio(xy)
    assert k.tolist() == [0, 2, 1, 3]
    direto = torch.cat([rede.sub_redes(xy[i:i + 1].unsqueeze(0), k[i:i + 1])[0] for i in range(4)])
    assert torch.equal(rede(xy), direto)
    pts, ka, kb, normal = pontos_interface(2, 10)
    assert pts.shape == (4, 10, 2) and normal.tolist() == [0, 0, 1, 1]


def test_modo_float64_mesmos_pontos_e_pesos():
    xy32 = pontos_colocacao(100, semente=3)
    with modo_float64():
        from parallel_pinns.physics import geometry
        xy64 = geometry.pontos_colocacao(100, semente=3)
        assert xy64.dtype == torch.float64 and torch.get_default_dtype() == torch.float64
    assert torch.equal(xy64.float(), xy32)
    assert torch.get_default_dtype() == torch.float32                    # estado restaurado


def test_amdahl_recupera_f():
    S = {p: amdahl(p, 0.6) for p in (1, 2, 3, 4, 6)}
    assert math.isclose(ajustar_amdahl(S), 0.6, rel_tol=1e-12)
