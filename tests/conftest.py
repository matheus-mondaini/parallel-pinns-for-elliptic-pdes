import json
import os
from pathlib import Path

import pytest
import torch

RAIZ = Path(__file__).resolve().parents[1]
EXPLORATORIO = RAIZ / "notebooks" / "exploratorio"


@pytest.fixture(autouse=True)
def torch_cpu_deterministico():
    # Mesmo estado em todos os testes: CPU, 1 thread, modo deterministico, float32
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield


@pytest.fixture
def resultado_exploratorio():
    # Le um .json de resultado da versao exploratoria (versionado em notebooks/exploratorio)
    def ler(nome):
        with open(EXPLORATORIO / nome) as arquivo:
            return json.load(arquivo)
    return ler
