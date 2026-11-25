# Configurações dos experimentos

Cada arquivo `.yaml` declara um experimento: o que muda entre as execuções (`grade`, `variantes` e `condicionais`) e o que fica fixo (`base`). O formato está em [`src/parallel_pinns/tracking/experiment.py`](../src/parallel_pinns/tracking/experiment.py).

```
python -m parallel_pinns.cli.run_experiment configs/exploratorio/h2_confirmatorio.yaml --listar
python -m parallel_pinns.cli.run_experiment configs/exploratorio/h2_confirmatorio.yaml
```

O segundo comando grava um `.json` por execução em `results/<nome>/` e um `manifesto.json` com a configuração, o commit do código, as versões das bibliotecas e a plataforma. Uma execução que já tem `.json` é reaproveitada, então um experimento interrompido continua de onde parou.

### `exploratorio/`: os experimentos da versão exploratória

| Arquivo | Experimento (notebook exploratório) | Execuções |
|---|---|---|
| `exploratorio/ddp_tost.yaml` | DDP $p = 2, 4$ × sequencial, TOST (seção 10) | 60 |
| `exploratorio/xpinn_distribuido.yaml` | XPINN distribuído por replicação de parâmetros (seção 11) | 20 |
| `exploratorio/escolha_lam_bc.yaml` | Ajuste de $\lambda_{BC}$ da monolítica (seção 13) | 12 |
| `exploratorio/h2_confirmatorio.yaml` | Teste confirmatório da H2, $n = 13$ e $15$ (seção 13) | 40 |
| `exploratorio/verificacao_cpu.yaml` | Monolítica de $n = 15$ em CPU (seção 13) | 10 |
| `exploratorio/ordem_somas.yaml` | Sequencial em `float64` (seção 13) | 20 |

Estes arquivos descrevem exatamente os experimentos da versão exploratória. Os nomes dos arquivos de resultado são os mesmos, e `tests/test_experimentos.py` confere que cada execução declarada aqui tem o seu `.json` em `notebooks/exploratorio/`. 
### `final/`: o núcleo da execução final do TCC

Grade da §13 do documento de definição: L0, P0, P1 e P2($n$) com $n \in \{5, 7, 11, 13\}$, 10 sementes (0 a 9), contorno fraco e orçamento fixo de 5000 iterações, sem platô.

| Arquivo | Configuração | Dispositivo | Execuções |
|---|---|---|---|
| `final/monolitica.yaml` | PINN monolítica 4 × 64 | GPU | 70 |
| `final/xpinn.yaml` | XPINN iso-parâmetro $M = 2$ e $M = 4$, $\lambda_I = 0{,}03$ | CPU | 140 |
| `final/ddp.yaml` | DDP $p = 4$ (`gloo`) e a referência sequencial $p = 1$ | CPU | 140 |

Os três podem rodar ao mesmo tempo, com os núcleos separados (só acurácia: os tempos de execuções simultâneas não entram em contas de speedup):

```
python -m parallel_pinns.cli.run_experiment configs/final/monolitica.yaml
python -m parallel_pinns.cli.run_experiment configs/final/xpinn.yaml --nucleos-cpu 2,4
python -m parallel_pinns.cli.run_experiment configs/final/ddp.yaml --nucleos-ddp 4
```
