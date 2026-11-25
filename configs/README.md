# Configurações dos experimentos

Cada arquivo `.yaml` declara um experimento:
- **`base`:** o que fica fixo em todas as execuções;
- **`grade`, `variantes` e `condicionais`:** o que muda entre elas;
- **`pasta`:** onde ficam os resultados. Experimentos com a mesma pasta compartilham execuções idênticas: o mesmo `.json` serve aos dois;
- **`exclusivo: true`:** marca uma medição de tempo, que roda uma execução por vez, sozinha na máquina.

O formato completo está em [`src/parallel_pinns/tracking/experiment.py`](../src/parallel_pinns/tracking/experiment.py).

Cada execução grava um `.json`, e cada experimento grava `manifesto_<nome>.json` com a configuração, o commit do código, as versões das bibliotecas e a plataforma. Uma execução que já tem `.json` é reaproveitada, então uma execução interrompida continua de onde parou.

## `final/`: a execução final do TCC

Um comando roda tudo e grava em `results/final/`:

```
python -m parallel_pinns.cli.run_all "configs/final/*.yaml"
```

A execução tem duas fases:
1. **Medições de tempo:** cada uma sozinha na máquina, com afinidade fixa.
2. **Acurácia:** tudo ao mesmo tempo. A GPU fica com uma tarefa por vez. A CPU começa com o DDP em 4 núcleos e as tarefas em 2; quando o DDP termina, os núcleos dele passam para as tarefas. Os tempos de execuções simultâneas não entram em contas de speedup.

Convenções comuns a todos os experimentos:
- contorno fraco;
- rede 4 × 64 (o XPINN usa a mesma capacidade total);
- orçamento fixo de 5000 iterações, sem platô;
- sementes 0 a 9, ou 0 a 19 nos testes de equivalência;
- os testes confirmatórios usam as sementes 20 a 29, e os ajustes, as 10 a 12.

As comparações de acurácia entre arquiteturas ficam no **mesmo dispositivo**, a CPU com 1 thread. A monolítica de referência é a execução sequencial ($p = 1$) do `nucleo_ddp`. A GPU fica com os estudos internos à monolítica: a varredura de $n$ e a formulação do contorno.

| Arquivo | O que responde (TCC 2) | Execuções |
|---|---|---|
| `tempo_ddp.yaml` | RQ1: speedup e Amdahl do DDP, $N_f \in \{2400, 4800, 9600\}$, $p \in \{1, 2, 3, 4, 6\}$, com o efeito do turbo | 18 |
| `tempo_xpinn_distribuido.yaml` | RQ1: speedup do XPINN distribuído e custo por ponto da arquitetura | 12 |
| `nucleo_ddp.yaml` | Núcleo: monolítica sequencial e DDP $p = 4$ em L0, P0, P1 e P2($n$), $n \in \{5, 7, 11, 13\}$ | 140 |
| `nucleo_xpinn.yaml` | Núcleo: XPINN iso-parâmetro $M = 2$ e $M = 4$ nos mesmos problemas; H2 abaixo do limiar ($n = 11$) | 140 |
| `h1_ddp_tost.yaml` | H1: TOST do DDP $p = 2$ e $p = 4$ contra o sequencial, P1, 20 sementes | 60 |
| `h1_ordem_somas.yaml` | H1: sequencial em `float64` (ruído de arredondamento do `float32`) | 20 |
| `h1_xpinn_distribuido.yaml` | H1: XPINN distribuído contra o sequencial (TOST) | 20 |
| `escolha_lam_i.yaml`, `escolha_lam_bc.yaml` | Verificação: escolha de $\lambda_I$ e $\lambda_{BC}$ (sementes 10 a 12) | 12 + 12 |
| `sens_contorno.yaml` | Verificação: contorno fraco, exato ($D$ produto) e transfinito em P2(11) | 30 |
| `rq2_fluxo.yaml` | RQ2: interface com e sem o termo de fluxo ($J_u$, $J_n$, $E_{bal}$) | 40 |
| `rq3_varredura.yaml`, `rq3_varredura_nf_fixo.yaml` | RQ3: limiar da monolítica, $n$ de 11 a 31, com controle de subamostragem | 50 + 6 |
| `h2_confirmatorio.yaml` | H2: teste confirmatório, $n = 13$ e $15$, monolítica em CPU (teste principal) e na GPU (réplica do protocolo exploratório) | 60 |
| **Total sem repetição** | | **570** |

Os valores de $\lambda_I = 0{,}03$ e $\lambda_{BC} = 1$ foram fixados antes, na versão exploratória. As escolhas refeitas aqui servem para conferir esses valores, e não para mudá-los. A verificação do pipeline (MMS, FDM, DST, determinismo e equivalência do DDP) está nos testes automáticos (`pytest`).

## `exploratorio/`: os experimentos da versão exploratória

| Arquivo | Experimento (notebook exploratório) | Execuções |
|---|---|---|
| `exploratorio/tempo_ddp.yaml` | Tempo e Amdahl do DDP (seção 10) | 18 |
| `exploratorio/tempo_xpinn_distribuido.yaml` | Tempo do XPINN distribuído (seção 11) | 12 |
| `exploratorio/ddp_tost.yaml` | DDP $p = 2, 4$ × sequencial, TOST (seção 10) | 60 |
| `exploratorio/xpinn_distribuido.yaml` | XPINN distribuído por replicação de parâmetros (seção 11) | 20 |
| `exploratorio/escolha_lam_bc.yaml` | Ajuste de $\lambda_{BC}$ da monolítica (seção 13) | 12 |
| `exploratorio/h2_confirmatorio.yaml` | Teste confirmatório da H2, $n = 13$ e $15$ (seção 13) | 40 |
| `exploratorio/verificacao_cpu.yaml` | Monolítica de $n = 15$ em CPU (seção 13) | 10 |
| `exploratorio/ordem_somas.yaml` | Sequencial em `float64` (seção 13) | 20 |

Estes arquivos descrevem exatamente os experimentos da versão exploratória, com os mesmos nomes de arquivo de resultado. O `tests/test_experimentos.py` confere que cada execução declarada aqui tem o seu `.json` em `notebooks/exploratorio/`. Para rodar um deles:

```
python -m parallel_pinns.cli.run_experiment configs/exploratorio/h2_confirmatorio.yaml
```
