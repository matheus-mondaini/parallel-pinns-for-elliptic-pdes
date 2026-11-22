"""Fila de treinos independentes: uma tarefa na GPU e uma por nucleo fisico livre na CPU, um .json por execucao.

So ACURACIA: tarefas simultaneas disputam memoria, entao os tempos delas nao entram em contas de speedup.
"""
import json
import os
import subprocess
import sys
import time

import psutil

from .distributed import ambiente_filho

PADRAO_TAREFA = dict(problema="P2", n=None, semente=0, N_f=2000, N_b=400, lam_bc=1.0, M=0, lam_i=1.0,
                     epocas_adam=3000, passos_lbfgs=100, sem_plato=True, float64=False, dispositivo="cpu")


def arquivo_tarefa(cfg):
    # Nome a partir das opcoes que diferem do padrao: a mesma configuracao sempre cai no mesmo arquivo (retomada)
    partes = [f"{k}{cfg[k]}" for k in sorted(cfg) if k in ("problema", "semente", "dispositivo") or cfg[k] != PADRAO_TAREFA[k]]
    return "tarefa_" + "_".join(partes).replace(".", "p") + ".json"


def rodar_fila(tarefas, nucleos_cpu=(2, 4, 6, 8, 10), nucleo_gpu=0, reaproveitar=True, pasta="."):
    # GPU: uma tarefa por vez, presa ao nucleo fisico 0. CPU: uma tarefa por nucleo fisico livre (1 thread cada).
    # reaproveitar: se o .json ja existe, nao roda de novo (retomada depois de uma interrupcao).
    cfgs = [dict(PADRAO_TAREFA, **t) for t in tarefas]
    caminho = lambda c: os.path.join(pasta, arquivo_tarefa(c))
    fila = {"cuda": [], "cpu": []}
    for i, c in enumerate(cfgs):
        if not (reaproveitar and os.path.exists(caminho(c))):
            fila[c["dispositivo"]].append(i)
    livres, ativos = {"cuda": [nucleo_gpu], "cpu": list(nucleos_cpu)}, []
    try:
        while any(fila.values()) or ativos:
            for d in ("cuda", "cpu"):
                while fila[d] and livres[d]:
                    i, nucleo = fila[d].pop(0), livres[d].pop(0)
                    saida = caminho(cfgs[i])
                    extra = {"CUDA_VISIBLE_DEVICES": ""} if d == "cpu" else {}
                    log = open(saida + ".log", "w")
                    proc = subprocess.Popen([sys.executable, "-m", "parallel_pinns.cli.run_task", json.dumps(cfgs[i]),
                                             saida], env=ambiente_filho(**extra), stdout=log, stderr=log)
                    psutil.Process(proc.pid).cpu_affinity([nucleo % os.cpu_count()])
                    ativos.append((i, proc, log, nucleo, d))
            time.sleep(1)
            for a in [a for a in ativos if a[1].poll() is not None]:
                i, proc, log, nucleo, d = a
                ativos.remove(a)
                log.close()
                livres[d].append(nucleo)
                saida = caminho(cfgs[i])
                if proc.returncode != 0:
                    with open(saida + ".log") as arquivo:
                        raise RuntimeError(f"tarefa falhou: {saida}\n" + arquivo.read()[-3000:])
                os.remove(saida + ".log")
    finally:
        for _, proc, log, _, _ in ativos:         # numa falha (ou interrupcao), nao deixa processos soltos
            proc.kill()
            proc.wait()
            log.close()
    resultados = []
    for c in cfgs:
        with open(caminho(c)) as arquivo:
            resultados.append(json.load(arquivo))
    return resultados
