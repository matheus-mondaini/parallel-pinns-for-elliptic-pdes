# Versão exploratória (notebook monolítico)

**Os números deste notebook são exploratórios.** Os números do TCC vêm da execução final, feita com o pacote em `src/`.

- `pinn_tcc_laplace_poisson_2d.ipynb` é o notebook de 107 células exatamente como gerou os resultados exploratórios: a execução completa e três re-execuções parciais, de 17 a 19/11/2026. O código está todo nas células, e o notebook se lê de cima a baixo, sem precisar abrir outros arquivos.
- Os `.json` são os resultados por execução que o notebook reaproveita em vez de treinar de novo: DDP, XPINN distribuído, teste confirmatório da H2, `float64` e a verificação em CPU.
- A versão congelada é a tag [`v0.1.0-exploratorio`](../../CHANGELOG.md). Ela nunca muda, e esta pasta é só uma cópia de leitura:

```
git checkout v0.1.0-exploratorio
```

**Para rodar aqui:** abra o notebook com esta pasta como diretório de trabalho. As células gravam `pinn_comum.py`, `pinn_xpinn.py`, `ddp_pinn.py` e `tarefa_pinn.py` nesta pasta. Esses arquivos são ignorados pelo git, e o notebook os usa para lançar o DDP e a fila de tarefas.

**Relação com `src/`:** cada função do notebook tem uma correspondente no pacote, com o mesmo nome (tabela no [README principal](../../README.md#da-versão-exploratória-para-o-pacote)). O teste `tests/test_regressao_v0.py` reconstrói o código desta versão a partir da tag e confere, bit a bit, que o pacote dá os mesmos resultados.
