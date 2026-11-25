# Changelog

Formato baseado em [Keep a Changelog 1.1.0](https://keepachangelog.com/pt-BR/1.1.0/). As versões seguem o [Versionamento Semântico 2.0.0](https://semver.org/lang/pt-BR/). Enquanto a versão for `0.y.z`, a interface ainda pode mudar.

## [Não lançado]

### Adicionado
- Pacote `parallel_pinns` em `src/` (*src layout*), com os módulos da arquitetura proposta no TCC (§3.2): `physics`, `models`, `training`, `analysis`, `references`, `statistics`, `parallel` e `utils`. O código é o do notebook, com os mesmos nomes de função.
- Pontos de entrada `python -m parallel_pinns.cli.train_parallel` (DDP e XPINN distribuído) e `python -m parallel_pinns.cli.run_task` (um treino por processo). Cada execução grava um `.json`, o que permite retomar.
- Testes `pytest`:
  - física e numérica: MMS, contorno, DST, convergência $h^2$ do FDM, contagem de parâmetros e Amdahl;
  - estatística: refaz os números das seções 10, 11 e 13 do notebook a partir dos `.json` versionados;
  - **regressão contra a tag exploratória**, bit a bit;
  - versão completa da regressão, com 5000 iterações (`pytest -m lento`).
- `pyproject.toml` (PEP 621), `CHANGELOG.md` e READMEs.
- Experimentos declarativos em `configs/`, um `.yaml` por experimento. Os seis arquivos descrevem os experimentos da versão exploratória.
- `python -m parallel_pinns.cli.run_experiment`, que grava em `results/<experimento>/` um `.json` por execução e um `manifesto.json` (`parallel_pinns.tracking`). O manifesto registra a configuração, o commit, as versões e a plataforma.
- `notebooks/01_resultados.ipynb`, que só lê os resultados e gera as tabelas e as figuras.
- `uv.lock`, com as versões exatas dos resultados: PyTorch 2.13 com CUDA 13.2, NumPy 2.4.6 e SciPy 1.16.3.
- README: tabela das diferenças em relação à proposta do TCC.

### Corrigido
- `README.md` estava em UTF-16. Agora está em UTF-8, o que permite construir o pacote.

### Mudado
- O notebook e os resultados exploratórios foram para `notebooks/exploratorio/`, sem alteração de conteúdo.
- As funções de estatística devolvem valores em vez de imprimir.
- `criar_problema` passa a recusar um benchmark desconhecido, em vez de devolver `None`.

## [0.1.0-exploratorio] - 2026-11-19

### Adicionado
- Notebook monolítico `pinn_tcc_laplace_poisson_2d.ipynb` (107 células), com:
  - PINN monolítica, DDP (`gloo`, CPU) e XPINN, com verificação estatística entre sementes;
  - teste confirmatório da H2, teste da ordem das somas (`float64`) e verificação em CPU.
- Resultados por execução (`.json`) que o notebook reaproveita.

[Não lançado]: https://github.com/matheus-mondaini/parallel-pinns-for-elliptic-pdes/compare/v0.1.0-exploratorio...HEAD
[0.1.0-exploratorio]: https://github.com/matheus-mondaini/parallel-pinns-for-elliptic-pdes/releases/tag/v0.1.0-exploratorio
