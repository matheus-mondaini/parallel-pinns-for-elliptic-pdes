"""Experimentos declarativos: um arquivo YAML em configs/ vira a lista de execucoes, um .json por execucao.

Formato do arquivo de configuracao::

    nome: h2_confirmatorio            # pasta dos resultados: <resultados>/<nome>/
    tipo: tarefas                     # tarefas (rodar_fila) ou ddp (rodar_ddp_lote)
    base: {problema: P2, sem_plato: true}
    grade:                            # produto cartesiano
      n: [13, 15]
      semente: [20, 21, 22]
    variantes:                        # cada variante e combinada com toda a grade
      - {dispositivo: cuda}
      - {M: 4, lam_i: 0.03}
    condicionais:                     # ajustes que dependem de outros valores
      - {quando: {n: 15}, usar: {N_f: 3719, N_b: 544}}

Cada pasta de resultados ganha um ``manifesto.json`` com a configuracao, o commit do codigo, as versoes das
bibliotecas e a plataforma: e o registro de proveniencia de cada numero (TCC, secao 3.6).
"""
import datetime
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
from pathlib import Path

import yaml

from ..parallel.distributed import nome_resultado, rodar_ddp_lote
from ..parallel.task_queue import PADRAO_TAREFA, arquivo_tarefa, rodar_fila

TIPOS = ("tarefas", "ddp")


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
    t = dict(t)
    return nome_resultado(t.pop("problema"), t.pop("p"), t.pop("semente", 0), t.pop("n", None), **t)


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifesto(cfg):
    import numpy
    import scipy
    import torch

    versoes = {}
    for pacote in ("parallel-pinns-elliptic-pdes",):
        try:
            versoes[pacote] = importlib.metadata.version(pacote)
        except importlib.metadata.PackageNotFoundError:
            versoes[pacote] = None
    return {
        "config": cfg,
        "commit": _git("rev-parse", "HEAD"),
        "alteracoes_nao_commitadas": bool(_git("status", "--porcelain", "--", "src", "configs")),
        "versoes": {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
                    "scipy": scipy.__version__, **versoes},
        "plataforma": {"sistema": platform.platform(), "processador": platform.processor(),
                       "nucleos_logicos": os.cpu_count(),
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }


def rodar_experimento(cfg, resultados="results", reaproveitar=True, nucleos_cpu=(2, 4, 6, 8, 10), nucleos_ddp=6):
    pasta = Path(resultados) / cfg["nome"]
    pasta.mkdir(parents=True, exist_ok=True)
    registro = manifesto(cfg)
    registro["inicio"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    execucoes = expandir(cfg)
    if cfg["tipo"] == "tarefas":
        saidas = rodar_fila(execucoes, nucleos_cpu=nucleos_cpu, reaproveitar=reaproveitar, pasta=str(pasta))
    else:
        lote = [dict({k: v for k, v in t.items() if k != "problema"}, nome=t["problema"], pasta=str(pasta),
                     reaproveitar=reaproveitar) for t in execucoes]
        saidas = rodar_ddp_lote(lote, nucleos=nucleos_ddp)
    registro["fim"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    registro["arquivos"] = [arquivo_resultado(cfg["tipo"], t) for t in execucoes]
    (pasta / "manifesto.json").write_text(json.dumps(registro, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return list(zip(execucoes, saidas))


def carregar_resultados(cfg, pasta):
    # Le os resultados de um experimento sem rodar nada. pasta: onde estao os .json
    # (results/<nome> depois de rodar_experimento, ou notebooks/exploratorio para a versao exploratoria)
    execucoes = expandir(cfg)
    caminhos = [Path(pasta) / arquivo_resultado(cfg["tipo"], t) for t in execucoes]
    faltando = [c.name for c in caminhos if not c.exists()]
    if faltando:
        raise FileNotFoundError(f"{len(faltando)} resultado(s) de '{cfg['nome']}' faltando em {pasta}: {faltando[:3]}...")
    return [(t, json.loads(c.read_text())) for t, c in zip(execucoes, caminhos)]
