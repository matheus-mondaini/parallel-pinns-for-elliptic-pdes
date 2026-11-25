# Parallel PINNs for elliptic PDEs

Código do TCC *Paralelismo em PINNs para resolução de EDPs elípticas*: PINN monolítica, paralelismo de dados (DDP) e decomposição de domínio (XPINN) para Laplace e Poisson em $(0,1)^2$.

## Duas versões, um repositório

| | Versão exploratória | Versão de engenharia (esta branch) |
|---|---|---|
| O que é | Notebook único, de 107 células, que se lê de cima a baixo | Pacote `src/parallel_pinns` com testes |
| Onde | Tag `v0.1.0-exploratorio` (imutável), com cópia de leitura em [`notebooks/exploratorio/`](notebooks/exploratorio/) | Branch principal (`main`) |
| Números | Exploratórios (17 a 19/11/2026) | Os da execução final do TCC |
| Para quem | Leitura linear do método e dos resultados | Reprodução, extensão e verificação automática |

A versão exploratória é uma **tag**, e não uma branch paralela. Uma branch é feita para ser mesclada de volta. Duas branches vivas acabam divergindo: um bug corrigido numa e esquecido na outra, e não se sabe mais de qual saiu cada número. A tag congela exatamente o código e os resultados exploratórios, e qualquer um volta a ela com:

```
git checkout v0.1.0-exploratorio
```

**A ponte entre as duas é um teste de regressão.** `tests/test_regressao_v0.py` reconstrói, a partir do notebook da tag, os módulos que ele gravava, roda casos curtos com eles e exige que o pacote dê os **mesmos bits**. Os casos cobrem:
- monolítica com os três tipos de contorno;
- XPINN;
- DDP com 2 processos;
- XPINN distribuído;
- `float64`.

A versão completa (`pytest -m lento`) refaz execuções de 5000 iterações e compara com os `.json` versionados da tag. Isso mostra que a refatoração não mudou a ciência.

### Práticas seguidas
- **Layout `src/`**, como recomendam o [Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) e o [Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/packaging-simple/). Os testes rodam contra o pacote, e não contra arquivos soltos na raiz.
- **Notebooks separados do código reutilizável** (`notebooks/` para exploração, `src/` para o pacote), como no [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) e em Wilson et al., [*Good enough practices in scientific computing*](https://doi.org/10.1371/journal.pcbi.1005510) (PLOS Comput. Biol., 2017).
- **Versões por tag anotada**, no [Versionamento Semântico](https://semver.org/lang/pt-BR/). O sufixo `-exploratorio` marca uma pré-versão. O histórico está no [`CHANGELOG.md`](CHANGELOG.md), no formato [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).
- **Um arquivo de resultado por execução**, com nome derivado da configuração. Uma execução interrompida retoma de onde parou, e cada número aponta para o `.json` que o gerou.

## Estrutura

```
src/parallel_pinns/
  physics/      benchmarks (pdes), laplaciano por autograd (operators), pontos (geometry), contorno exato
  models/       MLP (networks) e XPINN com sub-redes empilhadas (xpinn)
  training/     Adam + L-BFGS (optimizer), PINN sequencial/DDP (trainer), XPINN sequencial/distribuído
  analysis/     E_L2, E_H1, L_PDE, balanço global (metrics), DST-I e E_n/E_1 (fourier), saltos de interface
  references/   FDM de 5 pontos
  statistics/   resumo em log10, TOST, Wilcoxon, Hodges–Lehmann, regra de desfecho da H2
  parallel/     lançamento do DDP com gloo (distributed) e fila de treinos (task_queue)
  utils/        determinismo (seeds), float64 com os mesmos pontos e pesos (precision), Amdahl (metrics)
  tracking/     experimentos declarativos: YAML -> execuções -> .json + manifesto de proveniência
  cli/          python -m parallel_pinns.cli.run_experiment | train_parallel | run_task
configs/        um .yaml por experimento (ver configs/README.md)
results/final/  os 570 .json da execução final + manifesto_<experimento>.json
tests/          unidade, estatística sobre os resultados exploratórios, configs e regressão contra a tag
notebooks/
  01_resultados.ipynb          refaz as tabelas da versão exploratória a partir dos seus .json
  02_resultados_finais.ipynb   tabelas e figuras do TCC 2, só lendo results/final/
  figuras/                     as figuras do 02 em PDF, para o LaTeX
  exploratorio/                o notebook da tag e os seus resultados
pyproject.toml, uv.lock     dependências; o lock fixa as versões exatas dos resultados
```

Os módulos seguem a arquitetura proposta no TCC (§3.2). As diferenças em relação à proposta estão no fim deste arquivo.

## Uso

```
uv sync --extra dev                # ambiente travado pelo uv.lock (ou: pip install -e ".[dev]")
pytest                             # unidade + estatística + configs + regressão curta (~3 min)
pytest -m lento                    # regressão com 5000 iterações (dezenas de minutos)
```

Um experimento é um arquivo em `configs/`. Para rodá-lo:

```
python -m parallel_pinns.cli.run_experiment configs/exploratorio/h2_confirmatorio.yaml
```

O comando grava um `.json` por execução em `results/h2_confirmatorio/` e um `manifesto.json` com o commit, as versões e a plataforma. As tabelas e figuras saem de [`notebooks/01_resultados.ipynb`](notebooks/01_resultados.ipynb), que não treina nada.

```python
from parallel_pinns.physics import criar_problema
from parallel_pinns.training import treinar_pinn, treinar_xpinn
from parallel_pinns.analysis import avaliar

prob = criar_problema("P2", n=15)
rede, hist, tempo = treinar_xpinn(prob, M=4, lam_i=0.03, N_f=3719, N_b=544)
print(avaliar(rede, prob)["L2"])
```

O DDP é lançado com `parallel_pinns.parallel.rodar_ddp("P1", p=4, sem_plato=True)`, que faz o papel do `torchrun`. Treinos independentes vão para uma fila com `rodar_fila([...])`, com uma tarefa na GPU e uma por núcleo físico na CPU.

### Da versão exploratória para o pacote

Os nomes das funções foram mantidos, para que cada número do TCC possa ser rastreado até a função que o gerou.

| Notebook (`v0.1.0-exploratorio`) | Pacote |
|---|---|
| `criar_problema`, `phi`, benchmarks | `physics.pdes` |
| `laplaciano` | `physics.operators` |
| `pontos_colocacao`, `pontos_contorno`, `grade_interior`, `distancia_contorno` | `physics.geometry` |
| `ContornoExato`, `ContornoTransfinito` | `physics.boundary_conditions` |
| `criar_rede`, `n_parametros`, `largura_iso_parametro` | `models.networks` |
| `XPINN`, `agrupar_por_subdominio`, `pontos_interface` | `models.xpinn` |
| `otimizar` / `treinar_pinn` / `treinar_xpinn` | `training.optimizer` / `training.trainer` / `training.xpinn_trainer` |
| `avaliar`, `balanco_global`, `time_to_solution` | `analysis.metrics` |
| `coeficientes_seno`, `metricas_espectrais` | `analysis.fourier` |
| `saltos_interface`, `residuo_por_subdominio` | `analysis.interface` |
| `resolver_fdm` | `references.finite_difference` |
| `resumo_log`, `comparar_pareado`, `tost_log`, `hodges_lehmann`, `falhou`, `comparar_ordem` | `statistics.hypothesis_tests` (`comparar_ordem` virou `comparar_razao`) |
| `rodar_ddp`, `rodar_ddp_lote`, `porta_livre`, `cpu_do_rank` | `parallel.distributed` |
| `rodar_fila`, `arquivo_tarefa`, `PADRAO_TAREFA` | `parallel.task_queue` |
| `ajustar_amdahl` | `utils.metrics` |
| `ddp_pinn.py`, `tarefa_pinn.py` (gerados pelo notebook) | `cli.train_parallel`, `cli.run_task` |

## Plataforma

Os resultados exploratórios e a referência da regressão vêm de:
- **CPU:** Intel Xeon E5-2620 v3 (6 núcleos físicos);
- **GPU:** NVIDIA GTX 1660;
- **Software:** Windows 10, Python 3.11, PyTorch 2.13 (CUDA 13.2), NumPy 2.4 e SciPy 1.16.

O DDP roda em CPU, com `gloo`. Igualdade bit a bit só é garantida na mesma plataforma e com as mesmas versões.

## Execução final

`python -m parallel_pinns.cli.run_all "configs/final/*.yaml"` rodou as 570 execuções da [`configs/final/`](configs/README.md) em 25/11/2026 (01:29 às 19:42), no commit `540f1a4`, registrado em cada manifesto. Os resultados estão em `results/final/`, e as tabelas e figuras do TCC 2 saem de [`notebooks/02_resultados_finais.ipynb`](notebooks/02_resultados_finais.ipynb), que não treina nada. No texto, citar esse commit e a tag exploratória.

## Diferenças em relação à proposta do TCC (§3.2, §3.3 e §3.6)

| Proposta | Neste repositório | Por quê |
|---|---|---|
| DDP com NCCL em várias GPUs | DDP com `gloo` em CPU | A máquina tem uma GPU só |
| XPINN com troca só na interface, via MPI (`parallel/domain_decomposition.py`) | XPINN distribuído por replicação de parâmetros, com all-reduce `gloo` | O L-BFGS precisa de produtos internos globais; o esquema por interface fica como trabalho futuro (a RQ4 não é medida) |
| DeepXDE | PyTorch puro | Controle sobre o particionamento, o determinismo e o contorno fraco |
| Reamostragem adaptativa por resíduo (§3.3) | Não implementada | Fora do escopo das hipóteses testadas |
| Docker, `docker-compose` e devcontainer | Não incluídos | O ambiente fica fixado pelo `uv.lock` e pela plataforma registrada no manifesto; uma imagem só faria sentido para execução em cluster |
| Weights & Biases e MLflow (`tracking/`) | Manifesto local por experimento | Um `.json` por execução mais o manifesto (commit, versões, plataforma) versionados no git dão a mesma proveniência, sem serviço externo |
| FEM (`references/fem.py`) | Só o FDM de 5 pontos | A referência de erro é a solução analítica; o FDM só verifica o pipeline de métricas |
| `statistics/bootstrap.py`, `uncertainty.py` | Estatística em escala log10: IC $t$, TOST, Wilcoxon e Hodges–Lehmann | Os erros variam por ordens de grandeza entre sementes; testes pareados respondem às hipóteses |
| Média e desvio padrão aritméticos (§3.4) | Média geométrica e fator de dispersão | Mesmo motivo |
| `benchmarks/` como pacote, problema multiescala | Benchmarks L0, P0, P1, P2($n$) e L1($m$) em `physics/pdes.py` | O P2($n$) com $n$ variável é o problema de frequência alta da H2 |
| `scripts/`, `evaluation/`, `utils/plotting.py` e `loggers` | `cli/`, `analysis/` e o notebook de resultados | Mesma função, organizada dentro do pacote |
| FFT do erro em intervalos de época (`analysis/fourier.py`) | DST-I do campo final, com $E_n$ e $E_1$ ao longo do treino | Os coeficientes de seno são exatos para Dirichlet no quadrado |
