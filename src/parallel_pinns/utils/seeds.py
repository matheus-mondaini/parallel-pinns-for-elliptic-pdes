"""Configuracao de determinismo, a mesma do notebook exploratorio (celula de abertura)."""
import os
import warnings

import torch


def configurar_determinismo(dispositivo="cpu", threads=1):
    # Chamar antes de qualquer operacao na GPU: o cuBLAS so le CUBLAS_WORKSPACE_CONFIG na inicializacao.
    # Devolve o conjunto de avisos de operacoes sem versao deterministica (preenchido durante a execucao).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.set_default_device(dispositivo)
    torch.use_deterministic_algorithms(True, warn_only=True)
    avisos = set()
    mostrar = warnings.showwarning

    def registrar(message, category, filename, lineno, file=None, line=None):
        if "deterministic" in str(message):
            avisos.add(str(message).split("\n")[0][:300])
        else:
            mostrar(message, category, filename, lineno, file, line)

    warnings.showwarning = registrar
    return avisos
