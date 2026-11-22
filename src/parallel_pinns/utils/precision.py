"""Treino em float64 com os MESMOS pontos e pesos iniciais do float32 (teste da ordem das somas)."""
import contextlib

import torch

from ..models import networks
from ..physics import geometry


@contextlib.contextmanager
def modo_float64():
    # Pontos, rede (inicializacao) e grade de avaliacao sao gerados em float32 e convertidos: os valores float32 sao
    # exatos em float64. Assim o treino em float64 parte dos mesmos pontos e dos mesmos pesos; so a aritmetica muda.
    def em_float32(funcao):
        def envolvida(*a, **k):
            torch.set_default_dtype(torch.float32)
            try:
                r = funcao(*a, **k)
            finally:
                torch.set_default_dtype(torch.float64)
            return tuple(t.double() for t in r) if isinstance(r, tuple) else r.double()
        return envolvida

    alvos = [(geometry, "pontos_colocacao"), (geometry, "pontos_contorno"), (geometry, "grade_interior"),
             (networks, "criar_rede")]
    originais = [(modulo, nome, getattr(modulo, nome)) for modulo, nome in alvos]
    dtype_antes = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    for modulo, nome, funcao in originais:
        setattr(modulo, nome, em_float32(funcao))
    try:
        yield
    finally:
        for modulo, nome, funcao in originais:
            setattr(modulo, nome, funcao)
        torch.set_default_dtype(dtype_antes)
