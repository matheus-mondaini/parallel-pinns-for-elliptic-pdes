# Configurações dos experimentos

Cada arquivo `.yaml` declara um experimento: o que muda entre as execuções (`grade`, `variantes` e `condicionais`) e o que fica fixo (`base`). O formato está em [`src/parallel_pinns/tracking/experiment.py`](../src/parallel_pinns/tracking/experiment.py).

```
python -m parallel_pinns.cli.run_experiment configs/h2_confirmatorio.yaml --listar
python -m parallel_pinns.cli.run_experiment configs/h2_confirmatorio.yaml
```

O segundo comando grava um `.json` por execução em `results/<nome>/` e um `manifesto.json` com a configuração, o commit do código, as versões das bibliotecas e a plataforma. Uma execução que já tem `.json` é reaproveitada, então um experimento interrompido continua de onde parou.

| Arquivo | Experimento (notebook exploratório) | Execuções |
|---|---|---|
| `ddp_tost.yaml` | DDP $p = 2, 4$ × sequencial, TOST (seção 10) | 60 |
| `xpinn_distribuido.yaml` | XPINN distribuído por replicação de parâmetros (seção 11) | 20 |
| `escolha_lam_bc.yaml` | Ajuste de $\lambda_{BC}$ da monolítica (seção 13) | 12 |
| `h2_confirmatorio.yaml` | Teste confirmatório da H2, $n = 13$ e $15$ (seção 13) | 40 |
| `verificacao_cpu.yaml` | Monolítica de $n = 15$ em CPU (seção 13) | 10 |
| `ordem_somas.yaml` | Sequencial em `float64` (seção 13) | 20 |

Estes arquivos descrevem exatamente os experimentos da versão exploratória. Os nomes dos arquivos de resultado são os mesmos, e `tests/test_experimentos.py` confere que cada execução declarada aqui tem o seu `.json` em `notebooks/exploratorio/`. Os experimentos da execução final do TCC entram como novos arquivos neste formato.
