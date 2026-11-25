"""Regressao com o orcamento completo (5000 iteracoes): o pacote refaz execucoes da versao exploratoria e tem de
chegar aos MESMOS valores gravados nos .json versionados da tag. Lento (minutos por caso):

    pytest -m lento

So vale na mesma plataforma (CPU, PyTorch e bibliotecas numericas iguais): bits identicos entre maquinas
diferentes nao sao garantidos.
"""
import pytest

from parallel_pinns.parallel import arquivo_tarefa, nome_resultado, rodar_ddp, rodar_fila
from parallel_pinns.parallel.task_queue import PADRAO_TAREFA

pytestmark = pytest.mark.lento


@pytest.mark.parametrize("p", [1, 2])
def test_ddp_P1_semente0(tmp_path, resultado_exploratorio, p):
    ref = resultado_exploratorio(nome_resultado("P1", p, 0, sem_plato=True))
    r = rodar_ddp("P1", p=p, semente=0, sem_plato=True, pasta=str(tmp_path), reaproveitar=False)
    assert {k: r["metricas"][k] for k in ref["metricas"]} == ref["metricas"]   # + E_bal, J_u, J_n (novas)
    assert r["hist"]["loss"] == ref["hist"]["loss"]


@pytest.mark.parametrize("cfg", [
    dict(problema="P1", semente=0, float64=True),                     # teste da ordem das somas
    dict(problema="P2", n=13, semente=20, M=4, lam_i=0.03),            # teste confirmatorio da H2 (XPINN em CPU)
])
def test_tarefa(tmp_path, resultado_exploratorio, cfg):
    ref = resultado_exploratorio(arquivo_tarefa(dict(PADRAO_TAREFA, **cfg)))
    (r,) = rodar_fila([cfg], nucleos_cpu=(0,), reaproveitar=False, pasta=str(tmp_path))
    assert {k: r["metricas"][k] for k in ref["metricas"]} == ref["metricas"]   # + E_bal, J_u, J_n (novas)
    assert r["it_final"] == ref["it_final"] == 5000
