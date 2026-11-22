"""Validacao estatistica entre sementes (TCC, secao 3.4)."""
from .hypothesis_tests import (comparar_pareado, comparar_razao, desfecho_h2, falhou, hodges_lehmann, resumo_log,
                               tost_log)

__all__ = ["comparar_pareado", "comparar_razao", "desfecho_h2", "falhou", "hodges_lehmann", "resumo_log", "tost_log"]
