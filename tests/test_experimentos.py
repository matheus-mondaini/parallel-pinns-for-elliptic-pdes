"""Experimentos declarativos: expansao das configuracoes, nomes dos resultados e execucao de ponta a ponta."""
import json
from pathlib import Path

import pytest

from parallel_pinns.cli.run_experiment import main as run_experiment
from parallel_pinns.tracking import arquivo_resultado, carregar_config, carregar_resultados, expandir

RAIZ = Path(__file__).resolve().parents[1]
CONFIGS = sorted((RAIZ / "configs" / "exploratorio").glob("*.yaml"))
ESPERADO = {"ddp_tost": 60, "xpinn_distribuido": 20, "escolha_lam_bc": 12, "h2_confirmatorio": 40,
            "verificacao_cpu": 10, "ordem_somas": 20}


@pytest.mark.parametrize("caminho", CONFIGS, ids=lambda c: c.stem)
def test_config_descreve_os_resultados_exploratorios(caminho):
    # Cada execucao declarada tem o seu .json na versao exploratoria: a configuracao e o experimento que foi feito
    cfg = carregar_config(caminho)
    assert cfg["nome"] == caminho.stem
    resultados = carregar_resultados(cfg, RAIZ / "notebooks" / "exploratorio")
    assert len(resultados) == ESPERADO[cfg["nome"]]


def test_todas_as_configs_estao_cobertas():
    assert {c.stem for c in CONFIGS} == set(ESPERADO)


@pytest.mark.parametrize("arquivo,nome,total", [("monolitica", "final_monolitica", 70), ("xpinn", "final_xpinn", 140),
                                                 ("ddp", "final_ddp", 140)])
def test_grade_final(arquivo, nome, total):
    # Nucleo do documento de definicao: L0, P0, P1 e P2(n), n em {5, 7, 11, 13}, 10 sementes, orcamento fixo
    cfg = carregar_config(RAIZ / "configs" / "final" / f"{arquivo}.yaml")
    execucoes = expandir(cfg)
    assert cfg["nome"] == nome and len(execucoes) == total
    assert len({arquivo_resultado(cfg["tipo"], t) for t in execucoes}) == total      # um .json por execucao
    assert all(t["sem_plato"] for t in execucoes)
    assert {(t["problema"], t.get("n")) for t in execucoes} == {("L0", None), ("P0", None), ("P1", None), ("P2", 5),
                                                                 ("P2", 7), ("P2", 11), ("P2", 13)}


def test_expansao_com_variantes_e_condicionais():
    cfg = {"nome": "x", "tipo": "tarefas", "base": {"problema": "P2"}, "grade": {"n": [13, 15], "semente": [1, 2]},
           "variantes": [{}, {"M": 4}], "condicionais": [{"quando": {"n": 15}, "usar": {"N_f": 3719}}]}
    execucoes = expandir(cfg)
    assert len(execucoes) == 8
    assert all(t.get("N_f") == 3719 for t in execucoes if t["n"] == 15)
    assert all("N_f" not in t for t in execucoes if t["n"] == 13)
    assert arquivo_resultado("tarefas", execucoes[0]) == "tarefa_dispositivocpu_n13_problemaP2_semente1.json"


def test_opcao_desconhecida_e_recusada():
    with pytest.raises(ValueError):
        arquivo_resultado("tarefas", {"problema": "P1", "lr": 0.1})


def test_experimento_de_ponta_a_ponta(tmp_path):
    # Orcamento minimo: grava os .json e o manifesto com o commit; a segunda chamada reaproveita tudo
    cfg = tmp_path / "mini.yaml"
    cfg.write_text("nome: mini\ntipo: tarefas\nbase: {problema: P1, epocas_adam: 5, passos_lbfgs: 1}\n"
                   "grade: {semente: [0, 1]}\n", encoding="utf-8")
    run_experiment([str(cfg), "--resultados", str(tmp_path / "res")])
    pasta = tmp_path / "res" / "mini"
    manifesto = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    assert len(manifesto["arquivos"]) == 2 and all((pasta / a).exists() for a in manifesto["arquivos"])
    assert manifesto["config"]["nome"] == "mini" and "torch" in manifesto["versoes"]
    antes = [(pasta / a).stat().st_mtime_ns for a in manifesto["arquivos"]]
    run_experiment([str(cfg), "--resultados", str(tmp_path / "res")])
    assert [(pasta / a).stat().st_mtime_ns for a in manifesto["arquivos"]] == antes
