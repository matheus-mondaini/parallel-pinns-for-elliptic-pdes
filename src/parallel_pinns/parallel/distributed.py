"""Lancamento do DDP em CPU (backend gloo): p processos, um por nucleo fisico, um .json por execucao.

Faz o papel do torchrun: define RANK, WORLD_SIZE e MASTER_* e dispara ``parallel_pinns.cli.train_parallel``.
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

RAIZ_PACOTE = str(Path(__file__).resolve().parents[2])      # pasta que contem parallel_pinns (src/ ou site-packages)


def ambiente_filho(**extra):
    # Os processos filhos importam o pacote mesmo sem instalacao (pytest com pythonpath = src)
    caminho = os.pathsep.join(filter(None, [RAIZ_PACOTE, os.environ.get("PYTHONPATH")]))
    return dict(os.environ, PYTHONPATH=caminho, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", **extra)


def porta_livre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cpu_do_rank(rank):
    # Um processo por nucleo FISICO: no Windows, os processadores logicos 2k e 2k+1 sao o mesmo nucleo
    return [(2 * rank) % os.cpu_count()]


def nome_resultado(nome, p, semente, n=None, rotulo="", **opcoes):
    # Mesmo nome de arquivo da versao exploratoria: as mesmas opcoes sempre caem no mesmo .json (retomada)
    sufixo = "".join(f"_{k}{'' if v is True else v}" for k, v in sorted(opcoes.items()) if v is not False)
    return f"ddp_{nome}{'' if n is None else n}_p{p}_s{semente}{sufixo}{rotulo}.json"


def iniciar_ddp(nome, p=4, semente=0, n=None, afinidade=False, reaproveitar=True, rotulo="", pasta=".", **opcoes):
    # Dispara os p processos. p = 1: referencia sequencial (treinar_pinn sem DDP).
    # opcoes: contorno, N_f, N_b, epocas_adam, passos_lbfgs, M, lam_i (valores); sem_plato, sem_avaliacao (True/False).
    # afinidade: prende o rank r ao processador logico 2r (um processo por nucleo fisico). Usado nas medicoes de tempo.
    # reaproveitar: se o arquivo de resultado ja existe, nao roda de novo (retomada depois de uma interrupcao).
    # rotulo: sufixo do arquivo, para separar medicoes com as mesmas opcoes e condicoes diferentes.
    saida = os.path.join(pasta, nome_resultado(nome, p, semente, n, rotulo, **opcoes))
    if reaproveitar and os.path.exists(saida):
        return saida, []
    argumentos = ["-m", "parallel_pinns.cli.train_parallel", "--problema", nome, "--semente", str(semente),
                  "--saida", saida]
    if n is not None:
        argumentos += ["--n", str(n)]
    for k, v in opcoes.items():
        if v is True:
            argumentos.append(f"--{k}")
        elif v is not False:
            argumentos += [f"--{k}", str(v)]
    if p == 1:
        argumentos.append("--sequencial")
    porta, procs = porta_livre(), []
    for rank in range(p):
        ambiente = ambiente_filho(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(p), MASTER_ADDR="127.0.0.1",
                                  MASTER_PORT=str(porta), USE_LIBUV="0", CUDA_VISIBLE_DEVICES="")   # DDP so em CPU
        log = open(f"{saida}.r{rank}.log", "w")
        proc = subprocess.Popen([sys.executable] + argumentos, env=ambiente, stdout=log, stderr=log)
        if afinidade:
            psutil.Process(proc.pid).cpu_affinity(cpu_do_rank(rank))
        procs.append((proc, log))
    return saida, procs


def concluir_ddp(saida, procs):
    for proc, log in procs:
        proc.wait()
        log.close()
    if any(proc.returncode != 0 for proc, _ in procs):
        with open(f"{saida}.r0.log") as arquivo:
            raise RuntimeError("DDP falhou:\n" + arquivo.read()[-3000:])
    for rank in range(len(procs)):
        os.remove(f"{saida}.r{rank}.log")
    with open(saida) as arquivo:
        return json.load(arquivo)


def rodar_ddp(nome, p=4, semente=0, n=None, **opcoes):
    return concluir_ddp(*iniciar_ddp(nome, p, semente, n, **opcoes))


def iniciar_carga(cpus):
    # Processos que so giram num laco vazio, cada um preso a um processador logico: os nucleos ficam ativos
    # (frequencia de turbo com todos os nucleos ocupados), sem disputar memoria. Mede o efeito do turbo.
    procs = []
    for c in cpus:
        pr = subprocess.Popen([sys.executable, "-c", "while True: pass"])
        psutil.Process(pr.pid).cpu_affinity([c % os.cpu_count()])
        procs.append(pr)
    time.sleep(2)
    return procs


def parar_carga(procs):
    for pr in procs:
        pr.kill()
        pr.wait()


def rodar_ddp_lote(tarefas, nucleos=6):
    # Varias execucoes ao mesmo tempo, sem passar de `nucleos` processos (1 por nucleo fisico).
    # So para ACURACIA: execucoes simultaneas disputam memoria, entao os tempos delas nao entram em contas de speedup.
    saidas, ativos, fila = [None] * len(tarefas), [], list(enumerate(tarefas))
    while fila or ativos:
        while fila and sum(len(pr) for _, _, pr in ativos) + fila[0][1].get("p", 4) <= nucleos:
            i, t = fila.pop(0)
            ativos.append((i, *iniciar_ddp(**t)))
        time.sleep(0.5)
        for item in [a for a in ativos if all(proc.poll() is not None for proc, _ in a[2])]:
            ativos.remove(item)
            saidas[item[0]] = concluir_ddp(item[1], item[2])
    return saidas
