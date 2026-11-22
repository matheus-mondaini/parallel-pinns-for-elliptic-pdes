"""Regressao contra a versao exploratoria (tag v0.1.0-exploratorio): o pacote em src/ tem de dar os MESMOS bits.

A referencia (tests/data/referencia_v0.json) foi gerada por ``tests/regressao/gerar_referencia_v0.py``, que executa
o codigo reconstruido do notebook da tag, e nao uma copia dele. Igualdade exata, sem tolerancia: a refatoracao
nao pode mudar nenhuma conta.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from parallel_pinns.parallel import rodar_ddp, rodar_fila
from parallel_pinns.parallel.distributed import ambiente_filho

AQUI = Path(__file__).resolve().parent
REFERENCIA = json.loads((AQUI / "data" / "referencia_v0.json").read_text())
CURTO = dict(epocas_adam=30, passos_lbfgs=2, sem_plato=True)


@pytest.fixture(scope="module")
def casos_src(tmp_path_factory):
    # Processo novo, como na geracao da referencia
    saida = tmp_path_factory.mktemp("casos") / "casos.json"
    subprocess.run([sys.executable, str(AQUI / "regressao" / "casos.py"), "src", ".", str(saida)],
                   env=ambiente_filho(CUDA_VISIBLE_DEVICES=""), check=True)
    return json.loads(saida.read_text())


@pytest.mark.parametrize("nome", sorted(REFERENCIA["casos"]))
def test_treino_em_processo_igual_a_v0(casos_src, nome):
    assert casos_src[nome] == REFERENCIA["casos"][nome]


@pytest.mark.parametrize("nome,opcoes,p", [
    ("ddp_P1_p1_sequencial_s0", {}, 1),
    ("ddp_P1_p2_s0", {}, 2),
    ("xpinn_dist_P1_M2_p2_s0", {"M": 2}, 2),
])
def test_ddp_e_xpinn_distribuido_iguais_a_v0(tmp_path, nome, opcoes, p):
    r = rodar_ddp("P1", p=p, semente=0, pasta=str(tmp_path), reaproveitar=False, **CURTO, **opcoes)
    ref = REFERENCIA["ddp"][nome]
    assert r["metricas"] == ref["metricas"]
    assert r["hist"]["loss"] == ref["loss"]
    assert r["hist"]["it_final"] == ref["it_final"]


@pytest.mark.parametrize("nome,cfg", [
    ("tarefa_P1_float64_s0", dict(problema="P1", semente=0, float64=True)),
    ("tarefa_P2n11_xpinn_M4_lami0p03_s0", dict(problema="P2", n=11, semente=0, M=4, lam_i=0.03)),
])
def test_tarefa_isolada_igual_a_v0(tmp_path, nome, cfg):
    (r,) = rodar_fila([dict(cfg, **CURTO)], nucleos_cpu=(0,), reaproveitar=False, pasta=str(tmp_path))
    ref = REFERENCIA["tarefas"][nome]
    assert r["metricas"] == ref["metricas"]
    assert (r["it_final"], r["dtype"]) == (ref["it_final"], ref["dtype"])


def test_referencia_vem_da_tag():
    assert REFERENCIA["tag"] == "v0.1.0-exploratorio"
    commit = subprocess.run(["git", "rev-list", "-n", "1", REFERENCIA["tag"]], cwd=AQUI, capture_output=True,
                            text=True)
    if commit.returncode != 0:
        pytest.skip("tag nao disponivel neste clone (git fetch --tags)")
    assert commit.stdout.strip() == REFERENCIA["commit"]
