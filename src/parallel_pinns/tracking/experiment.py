"""Experimentos declarativos: um arquivo YAML em configs/ vira a lista de execucoes, um .json por execucao.

Formato do arquivo de configuracao::

    nome: h2_confirmatorio            # identifica o experimento (e o manifesto)
    pasta: final                      # opcional: pasta dos resultados (padrao: o nome). Experimentos com a mesma
                                      # pasta compartilham execucoes identicas (o mesmo .json serve aos dois)
    tipo: tarefas                     # tarefas (rodar_fila) ou ddp (rodar_ddp_lote)
    exclusivo: false                  # true: medicao de tempo, uma execucao por vez e sozinha na maquina
    base: {problema: P2, sem_plato: true}
    grade:                            # produto cartesiano
      n: [13, 15]
      semente: [20, 21, 22]
    variantes:                        # cada variante e combinada com toda a grade
      - {dispositivo: cuda}
      - {M: 4, lam_i: 0.03}
    condicionais:                     # ajustes que dependem de outros valores
      - {quando: {n: 15}, usar: {N_f: 3719, N_b: 544}}

Num experimento ddp, ``afinidade: true`` prende cada rank a um nucleo fisico, ``rotulo`` entra no nome do arquivo
e ``carga: true`` ocupa os outros nucleos com lacos vazios durante a execucao (efeito do turbo).

Cada experimento grava ``manifesto_<nome>.json`` na pasta dos resultados, com a configuracao, o commit do codigo,
as versoes das bibliotecas e a plataforma: o registro de proveniencia de cada numero (TCC, secao 3.6).
"""
import datetime
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import threading
from pathlib import Path

import yaml

from ..parallel.distributed import (concluir_ddp, cpu_do_rank, iniciar_carga, iniciar_ddp, nome_resultado,
                                    parar_carga, rodar_ddp_lote)
from ..parallel.task_queue import PADRAO_TAREFA, arquivo_tarefa, rodar_fila

TIPOS = ("tarefas", "ddp")
OPCOES_DE_LANCAMENTO = ("afinidade", "rotulo", "carga")         # ddp: mudam como roda, nao o que e calculado


def carregar_config(caminho):
    with open(caminho, encoding="utf-8") as arquivo:
        cfg = yaml.safe_load(arquivo)
    if cfg.get("tipo") not in TIPOS:
        raise ValueError(f"{caminho}: 'tipo' deve ser um de {TIPOS}")
    if not cfg.get("nome"):
        raise ValueError(f"{caminho}: falta 'nome'")
    return cfg


def expandir(cfg):
    # Lista de execucoes, na ordem: variante, depois a grade (sem repeticoes)
    grade = cfg.get("grade") or {}
    execucoes, vistas = [], set()
    for variante in cfg.get("variantes") or [{}]:
        for valores in itertools.product(*grade.values()):
            t = {**(cfg.get("base") or {}), **dict(zip(grade, valores)), **variante}
            for regra in cfg.get("condicionais") or []:
                if all(t.get(k) == v for k, v in regra["quando"].items()):
                    t.update(regra["usar"])
            chave = json.dumps(t, sort_keys=True)
            if chave not in vistas:
                vistas.add(chave)
                execucoes.append(t)
    return execucoes


def arquivo_resultado(tipo, t):
    # O mesmo nome que a versao exploratoria usava: as mesmas opcoes sempre caem no mesmo .json
    if tipo == "tarefas":
        desconhecidas = set(t) - set(PADRAO_TAREFA)
        if desconhecidas:
            raise ValueError(f"opcoes desconhecidas para uma tarefa: {sorted(desconhecidas)}")
        return arquivo_tarefa(dict(PADRAO_TAREFA, **t))
    t = {k: v for k, v in t.items() if k not in ("afinidade", "carga")}
    return nome_resultado(t.pop("problema"), t.pop("p"), t.pop("semente", 0), t.pop("n", None),
                          rotulo=t.pop("rotulo", ""), **t)


def pasta_resultados(cfg, resultados):
    return Path(resultados) / cfg.get("pasta", cfg["nome"])


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifesto(cfg):
    import numpy
    import scipy
    import torch

    from .. import __version__

    return {
        "config": cfg,
        "commit": _git("rev-parse", "HEAD"),
        "alteracoes_nao_commitadas": bool(_git("status", "--porcelain", "--", "src", "configs")),
        "versoes": {"parallel_pinns": __version__, "python": platform.python_version(), "torch": torch.__version__,
                    "numpy": numpy.__version__, "scipy": scipy.__version__},
        "plataforma": {"sistema": platform.platform(), "processador": platform.processor(),
                       "nucleos_logicos": os.cpu_count(),
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }


def _lote_ddp(execucoes, pasta, reaproveitar):
    return [dict({k: v for k, v in t.items() if k not in ("problema", "carga")}, nome=t["problema"], pasta=str(pasta),
                 reaproveitar=reaproveitar) for t in execucoes]


def _rodar_exclusivo(cfg, execucoes, pasta, reaproveitar):
    # Medicao de tempo: uma execucao por vez, nada mais rodando (a nao ser a carga, quando pedida)
    if cfg["tipo"] == "tarefas":
        return [rodar_fila([t], nucleos_cpu=(0,), reaproveitar=reaproveitar, pasta=str(pasta))[0] for t in execucoes]
    saidas = []
    for t, lancamento in zip(execucoes, _lote_ddp(execucoes, pasta, reaproveitar)):
        carga = iniciar_carga([cpu_do_rank(k)[0] for k in range(t["p"], 6)]) if t.get("carga") else []
        try:
            saidas.append(concluir_ddp(*iniciar_ddp(**lancamento)))
        finally:
            parar_carga(carga)
    return saidas


def _gravar_manifesto(cfg, pasta, registro, execucoes):
    registro["fim"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    registro["arquivos"] = [arquivo_resultado(cfg["tipo"], t) for t in execucoes]
    destino = pasta / f"manifesto_{cfg['nome']}.json"
    destino.write_text(json.dumps(registro, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def rodar_experimento(cfg, resultados="results", reaproveitar=True, nucleos_cpu=(2, 4, 6, 8, 10), nucleos_ddp=6):
    pasta = pasta_resultados(cfg, resultados)
    pasta.mkdir(parents=True, exist_ok=True)
    registro = manifesto(cfg)
    registro["inicio"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    execucoes = expandir(cfg)
    if cfg.get("exclusivo"):
        saidas = _rodar_exclusivo(cfg, execucoes, pasta, reaproveitar)
    elif cfg["tipo"] == "tarefas":
        saidas = rodar_fila(execucoes, nucleos_cpu=nucleos_cpu, reaproveitar=reaproveitar, pasta=str(pasta))
    else:
        saidas = rodar_ddp_lote(_lote_ddp(execucoes, pasta, reaproveitar), nucleos=nucleos_ddp)
    _gravar_manifesto(cfg, pasta, registro, execucoes)
    return list(zip(execucoes, saidas))


def rodar_varios(cfgs, resultados="results", reaproveitar=True, nucleos_cpu=(2, 4), nucleos_ddp=4,
                 nucleos_apos_ddp=(6, 8, 10), registrar=print):
    # Execucao unica de varios experimentos: primeiro as medicoes de tempo (exclusivas, uma por vez, sozinhas);
    # depois todas as execucoes de acuracia ao mesmo tempo: tarefas na fila GPU + CPU e o DDP numa linha paralela.
    # Quando o DDP termina, os nucleos dele (nucleos_apos_ddp) passam para a fila de tarefas.
    # Execucoes iguais de experimentos diferentes com a mesma pasta rodam uma vez so.
    for cfg in [c for c in cfgs if c.get("exclusivo")]:
        registrar(f"[{datetime.datetime.now():%H:%M}] tempo: {cfg['nome']} ({len(expandir(cfg))} execucoes, sozinhas)")
        rodar_experimento(cfg, resultados, reaproveitar)

    resto = [c for c in cfgs if not c.get("exclusivo")]
    inicio = {c["nome"]: dict(manifesto(c), inicio=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
              for c in resto}
    por_pasta = {}
    for c in resto:
        pasta = pasta_resultados(c, resultados)
        pasta.mkdir(parents=True, exist_ok=True)
        for t in expandir(c):
            por_pasta.setdefault((c["tipo"], str(pasta)), {})[arquivo_resultado(c["tipo"], t)] = t
    tarefas = [(pasta, list(ts.values())) for (tipo, pasta), ts in por_pasta.items() if tipo == "tarefas"]
    ddps = [(pasta, list(ts.values())) for (tipo, pasta), ts in por_pasta.items() if tipo == "ddp"]
    registrar(f"[{datetime.datetime.now():%H:%M}] acuracia: {sum(len(t) for _, t in tarefas)} tarefas e "
              f"{sum(len(t) for _, t in ddps)} execucoes DDP, ao mesmo tempo")

    erros, extra = [], []

    def linha_ddp():
        try:
            for pasta, ts in ddps:
                rodar_ddp_lote(_lote_ddp(ts, pasta, reaproveitar), nucleos=nucleos_ddp)
            registrar(f"[{datetime.datetime.now():%H:%M}] DDP concluido; nucleos {list(nucleos_apos_ddp)} para as tarefas")
        except Exception as erro:                     # noqa: BLE001 - relatado depois que a outra linha termina
            erros.append(erro)
        finally:
            extra.extend(nucleos_apos_ddp)

    fio = threading.Thread(target=linha_ddp)
    fio.start()
    try:
        for pasta, ts in tarefas:
            rodar_fila(ts, nucleos_cpu=nucleos_cpu, reaproveitar=reaproveitar, pasta=pasta, nucleos_extra=extra)
    finally:
        fio.join()
    if erros:
        raise erros[0]
    for c in resto:
        _gravar_manifesto(c, pasta_resultados(c, resultados), inicio[c["nome"]], expandir(c))
    registrar(f"[{datetime.datetime.now():%H:%M}] concluido")


def carregar_resultados(cfg, pasta=None, resultados="results"):
    # Le os resultados de um experimento sem rodar nada. pasta: onde estao os .json (padrao: a pasta do experimento
    # em `resultados`; notebooks/exploratorio para a versao exploratoria)
    pasta = Path(pasta) if pasta is not None else pasta_resultados(cfg, resultados)
    execucoes = expandir(cfg)
    caminhos = [pasta / arquivo_resultado(cfg["tipo"], t) for t in execucoes]
    faltando = [c.name for c in caminhos if not c.exists()]
    if faltando:
        raise FileNotFoundError(f"{len(faltando)} resultado(s) de '{cfg['nome']}' faltando em {pasta}: {faltando[:3]}...")
    return [(t, json.loads(c.read_text())) for t, c in zip(execucoes, caminhos)]
