"""O modulo de estatistica, aplicado aos .json versionados da versao exploratoria, reproduz os numeros do notebook.

Os valores esperados sao os impressos no notebook da tag (secoes 10, 11 e 13), com os digitos que ele mostra.
Os nomes dos arquivos vem de ``arquivo_tarefa`` e ``nome_resultado`` do pacote: o teste confere tambem que o
pacote encontra os resultados antigos (retomada compativel).
"""
import numpy as np
import pytest

from parallel_pinns.parallel import arquivo_tarefa, nome_resultado
from parallel_pinns.parallel.task_queue import PADRAO_TAREFA
from parallel_pinns.statistics import comparar_razao, desfecho_h2, falhou, hodges_lehmann, tost_log

SEMENTES_DDP = range(20)
SEMENTES_H2 = range(20, 30)


def l2_ddp(ler, p, **opcoes):
    return [ler(nome_resultado("P1", p, s, sem_plato=True, **opcoes))["metricas"]["L2"] for s in SEMENTES_DDP]


def tarefas(ler, sementes, **cfg):
    return [ler(arquivo_tarefa(dict(PADRAO_TAREFA, semente=s, **cfg)))["metricas"] for s in sementes]


def config_h2(n):
    return dict(N_f=2000, N_b=400) if n <= 13 else dict(N_f=3719, N_b=544)


def test_tost_do_ddp_secao_10(resultado_exploratorio):
    seq = l2_ddp(resultado_exploratorio, 1)
    for p, esperado in {2: ("0.858", "0.675", "1.092", "0.31"), 4: ("0.800", "0.634", "1.009", "0.50")}.items():
        r, (a, b), pv = tost_log(l2_ddp(resultado_exploratorio, p), seq)
        assert (f"{r:.3f}", f"{a:.3f}", f"{b:.3f}", f"{pv:.2f}") == esperado
        assert pv > 0.025                                  # equivalencia nao demonstrada (Bonferroni)


def test_ordem_das_somas_secao_13(resultado_exploratorio):
    seq32 = l2_ddp(resultado_exploratorio, 1)
    seq64 = [m["L2"] for m in tarefas(resultado_exploratorio, SEMENTES_DDP, problema="P1", float64=True)]
    ddp4 = l2_ddp(resultado_exploratorio, 4)
    r = comparar_razao(seq64, seq32)
    assert (round(r["razao"], 3), round(r["hodges_lehmann"], 3), f"{r['p_wilcoxon']:.3g}") == (0.729, 0.816, "0.000105")
    assert [round(v, 3) for v in r["ic90"]] == [0.586, 0.905]
    r = comparar_razao(ddp4, seq64)
    assert (round(r["razao"], 3), round(r["hodges_lehmann"], 3), f"{r['p_wilcoxon']:.3g}") == (1.098, 1.081, "0.0192")
    assert r["ic90"][0] > 1                                # "outro caso": o IC nao contem 1
    r = comparar_razao(ddp4, seq32)                        # DDP p = 4 entre os dois sequenciais
    assert (round(r["hodges_lehmann"], 3), f"{r['p_wilcoxon']:.3g}") == (0.899, "0.00944")


@pytest.mark.parametrize("n,hl,p,desfecho,falhas_mono,falhas_xpinn", [
    (13, 0.189, "0.0645", "sem diferença detectável", 5, 2),
    (15, 0.082, "0.00195", "H2 confirmada", 8, 0),
])
def test_teste_confirmatorio_h2(resultado_exploratorio, n, hl, p, desfecho, falhas_mono, falhas_xpinn):
    mono = tarefas(resultado_exploratorio, SEMENTES_H2, n=n, dispositivo="cuda", **config_h2(n))
    xpinn = tarefas(resultado_exploratorio, SEMENTES_H2, n=n, M=4, lam_i=0.03, **config_h2(n))
    r = comparar_razao([m["L2"] for m in xpinn], [m["L2"] for m in mono])
    assert (round(r["hodges_lehmann"], 3), f"{r['p_wilcoxon']:.3g}") == (hl, p)
    assert desfecho_h2(r["p_wilcoxon"], np.log10(r["hodges_lehmann"]), alfa=0.05 / 2) == desfecho
    assert (sum(map(falhou, mono)), sum(map(falhou, xpinn))) == (falhas_mono, falhas_xpinn)


def test_verificacao_monolitica_em_cpu(resultado_exploratorio):
    cpu = tarefas(resultado_exploratorio, SEMENTES_H2, n=15, **config_h2(15))
    xpinn = tarefas(resultado_exploratorio, SEMENTES_H2, n=15, M=4, lam_i=0.03, **config_h2(15))
    r = comparar_razao([m["L2"] for m in xpinn], [m["L2"] for m in cpu])
    assert (round(r["hodges_lehmann"], 3), f"{r['p_wilcoxon']:.3g}", r["pior_em"]) == (0.094, "0.00195", 0)
    assert sum(map(falhou, cpu)) == 6


def test_xpinn_distribuido_equivalente_secao_11(resultado_exploratorio):
    ler = lambda p, s: resultado_exploratorio(nome_resultado("P1", p, s, M=2, sem_plato=True))["metricas"]["L2"]
    r, (a, b), pv = tost_log([ler(4, s) for s in range(10)], [ler(1, s) for s in range(10)])
    assert (round(r, 2), round(a, 2), round(b, 2)) == (1.01, 0.96, 1.06)
    assert f"{pv:.2g}" == "1.3e-05"


def test_hodges_lehmann_valores_conhecidos():
    assert hodges_lehmann([1.0]) == 1.0
    assert hodges_lehmann([0.0, 2.0]) == 1.0                 # medias de Walsh: 0, 1, 2
    assert hodges_lehmann([0.0, 0.0, 0.0, 9.0]) == 0.0       # robusto a um valor extremo (a media seria 2,25)
