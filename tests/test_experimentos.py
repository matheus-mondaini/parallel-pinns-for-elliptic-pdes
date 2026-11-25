"""Experimentos declarativos: expansao das configuracoes, nomes dos resultados e execucao de ponta a ponta."""
import json
from pathlib import Path

import pytest

from parallel_pinns.cli.run_all import main as run_all
from parallel_pinns.cli.run_experiment import main as run_experiment
from parallel_pinns.tracking import arquivo_resultado, carregar_config, carregar_resultados, expandir

RAIZ = Path(__file__).resolve().parents[1]
CONFIGS = sorted((RAIZ / "configs" / "exploratorio").glob("*.yaml"))
ESPERADO = {"ddp_tost": 60, "xpinn_distribuido": 20, "escolha_lam_bc": 12, "h2_confirmatorio": 40,
            "verificacao_cpu": 10, "ordem_somas": 20, "tempo_ddp": 18,
            "tempo_xpinn_distribuido": 12}


@pytest.mark.parametrize("caminho", CONFIGS, ids=lambda c: c.stem)
def test_config_descreve_os_resultados_exploratorios(caminho):
    # Cada execucao declarada tem o seu .json na versao exploratoria: a configuracao e o experimento que foi feito
    cfg = carregar_config(caminho)
    assert cfg["nome"] == caminho.stem
    resultados = carregar_resultados(cfg, RAIZ / "notebooks" / "exploratorio")
    assert len(resultados) == ESPERADO[cfg["nome"]]


def test_todas_as_configs_estao_cobertas():
    assert {c.stem for c in CONFIGS} == set(ESPERADO)


FINAL = {"nucleo_ddp": 140, "nucleo_xpinn": 140, "h1_ddp_tost": 60, "h1_ordem_somas": 20, "h1_xpinn_distribuido": 20,
         "escolha_lam_i": 12, "escolha_lam_bc": 12, "sens_contorno": 30, "rq2_fluxo": 40, "rq3_varredura": 50,
         "rq3_varredura_nf_fixo": 6, "h2_confirmatorio": 60, "tempo_ddp": 18, "tempo_xpinn_distribuido": 12}


def test_execucao_final():
    cfgs = {c.stem: carregar_config(c) for c in sorted((RAIZ / "configs" / "final").glob("*.yaml"))}
    assert {k: len(expandir(c)) for k, c in cfgs.items()} == FINAL
    assert all(c["pasta"] == "final" and c["nome"] == "final_" + k for k, c in cfgs.items())
    # Nucleo do documento de definicao: L0, P0, P1 e P2(n), n em {5, 7, 11, 13}, 10 sementes, orcamento fixo
    for k in ("nucleo_ddp", "nucleo_xpinn"):
        execucoes = expandir(cfgs[k])
        assert {(t["problema"], t.get("n")) for t in execucoes} == {("L0", None), ("P0", None), ("P1", None),
                                                                     ("P2", 5), ("P2", 7), ("P2", 11), ("P2", 13)}
        assert all(t["sem_plato"] for t in execucoes) and {t["semente"] for t in execucoes} == set(range(10))
    # As medicoes de tempo da execucao final sao as mesmas da exploratoria
    for k in ("tempo_ddp", "tempo_xpinn_distribuido"):
        exploratoria = carregar_config(RAIZ / "configs" / "exploratorio" / f"{k}.yaml")
        assert expandir(cfgs[k]) == expandir(exploratoria) and cfgs[k]["exclusivo"]


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
    manifesto = json.loads((pasta / "manifesto_mini.json").read_text(encoding="utf-8"))
    assert len(manifesto["arquivos"]) == 2 and all((pasta / a).exists() for a in manifesto["arquivos"])
    assert manifesto["config"]["nome"] == "mini" and "torch" in manifesto["versoes"]
    antes = [(pasta / a).stat().st_mtime_ns for a in manifesto["arquivos"]]
    run_experiment([str(cfg), "--resultados", str(tmp_path / "res")])
    assert [(pasta / a).stat().st_mtime_ns for a in manifesto["arquivos"]] == antes


def test_execucao_unica_com_tempo_tarefas_e_ddp(tmp_path):
    # run_all: a medicao de tempo (exclusiva, com carga) primeiro; depois tarefas e DDP ao mesmo tempo, na mesma pasta.
    # A execucao que aparece em dois experimentos roda uma vez so.
    curto = "epocas_adam: 5, passos_lbfgs: 1, sem_plato: true"
    yamls = {
        "a": ["nome: a", "pasta: comum", "tipo: tarefas", f"base: {{problema: P1, {curto}}}", "grade: {semente: [0, 1]}"],
        "b": ["nome: b", "pasta: comum", "tipo: tarefas", f"base: {{problema: P1, {curto}}}", "grade: {semente: [1, 2]}"],
        "c": ["nome: c", "pasta: comum", "tipo: ddp", f"base: {{problema: P1, {curto}}}", "grade: {p: [1, 2]}"],
        "t": ["nome: t", "pasta: comum", "tipo: ddp", "exclusivo: true",
              f"base: {{problema: P1, {curto}, sem_avaliacao: true, afinidade: true, rotulo: _afin}}",
              "variantes: [{p: 2}, {p: 1, carga: true, rotulo: _afin_carga}]"],
    }
    for nome, linhas in yamls.items():
        (tmp_path / f"{nome}.yaml").write_text(chr(10).join(linhas) + chr(10), encoding="utf-8")
    run_all([str(tmp_path / "*.yaml"), "--resultados", str(tmp_path / "res"), "--nucleos-cpu", "2"])
    pasta = tmp_path / "res" / "comum"
    for nome in "abct":
        cfg = carregar_config(tmp_path / f"{nome}.yaml")
        assert len(carregar_resultados(cfg, resultados=tmp_path / "res")) == len(expandir(cfg))
        assert (pasta / f"manifesto_{nome}.json").exists()
    assert len(list(pasta.glob("tarefa_*.json"))) == 3                       # semente 1 compartilhada
    assert (pasta / "ddp_P1_p1_s0_epocas_adam5_passos_lbfgs1_sem_avaliacao_sem_plato_afin_carga.json").exists()
