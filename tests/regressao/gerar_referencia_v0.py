"""Gera tests/data/referencia_v0.json executando o codigo da TAG exploratoria, e nao uma copia dele.

    python tests/regressao/gerar_referencia_v0.py [tag]

1. Le o notebook da tag com ``git show`` e reconstroi os modulos que ele gravava (pinn_comum.py, pinn_xpinn.py,
   ddp_pinn.py e tarefa_pinn.py), do mesmo jeito que as celulas do notebook faziam.
2. Roda os casos curtos de ``casos.py`` com esses modulos (in-process) e os scripts ``ddp_pinn.py`` e
   ``tarefa_pinn.py`` da tag em processos separados (DDP com 2 processos gloo, XPINN distribuido, float64).
3. Grava os resultados. O teste ``tests/test_regressao_v0.py`` roda os MESMOS casos com o pacote em src/ e exige
   igualdade bit a bit.

So precisa rodar de novo se os casos mudarem; a referencia fica versionada.
"""
import ast
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_NB = "pinn_tcc_laplace_poisson_2d.ipynb"
CURTO_CLI = ["--epocas_adam", "30", "--passos_lbfgs", "2", "--sem_plato"]
CASOS_DDP = {        # nome: (argumentos extras, numero de processos)
    "ddp_P1_p1_sequencial_s0": ([], 1),
    "ddp_P1_p2_s0": ([], 2),
    "xpinn_dist_P1_M2_p2_s0": (["--M", "2"], 2),
}
CURTO_TAREFA = dict(epocas_adam=30, passos_lbfgs=2, sem_plato=True)
CASOS_TAREFA = {
    "tarefa_P1_float64_s0": dict(problema="P1", semente=0, float64=True),
    "tarefa_P2n11_xpinn_M4_lami0p03_s0": dict(problema="P2", n=11, semente=0, M=4, lam_i=0.03),
}


def git(*args):
    return subprocess.run(["git", *args], cwd=RAIZ, check=True, capture_output=True).stdout


def fontes_do_notebook(nb):
    # Ultima definicao de cada funcao/classe (ordem de execucao) e a celula de cada uma
    celulas = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    defs, constantes = {}, {}
    for texto in celulas:
        try:
            arvore = ast.parse(texto)
        except SyntaxError:                       # celulas com %%writefile
            continue
        for no in arvore.body:
            if isinstance(no, (ast.FunctionDef, ast.ClassDef)):
                defs[no.name] = (ast.get_source_segment(texto, no) + "\n", texto, isinstance(no, ast.ClassDef))
            elif isinstance(no, ast.Assign) and isinstance(no.value, (ast.Constant, ast.List)):
                for alvo in no.targets:
                    if isinstance(alvo, ast.Name):
                        constantes[alvo.id] = no.value
    return celulas, defs, constantes


def reconstruir_modulos(nb, pasta):
    celulas, defs, constantes = fontes_do_notebook(nb)
    cabecalho = constantes["CABECALHO"].value
    nomes = [e.id for e in constantes["OBJETOS_COMUNS"].elts]
    # Funcoes: inspect.getsource. Classes: a celula inteira em que o metodo forward foi definido (como no notebook).
    fonte = lambda nome: defs[nome][1] + "\n" if defs[nome][2] else defs[nome][0]
    comum = "# Gerado automaticamente a partir do notebook. Nao editar a mao.\n" + cabecalho
    comum += "".join("\n\n" + fonte(nome) for nome in nomes)
    xpinn = ("# Gerado automaticamente a partir do notebook. Nao editar a mao.\n"
             "import math\n\nimport numpy as np\nimport torch\nimport torch.distributed as dist\n\nfrom pinn_comum import *\n"
             "\n\n" + defs["XPINN"][1])
    ddp = next(c for c in celulas if c.startswith("%%writefile ddp_pinn.py")).split("\n", 1)[1]
    tarefa = constantes["TAREFA_SCRIPT"].value
    arquivos = {"pinn_comum.py": comum, "pinn_xpinn.py": xpinn, "ddp_pinn.py": ddp, "tarefa_pinn.py": tarefa}
    for nome, texto in arquivos.items():
        Path(pasta, nome).write_text(texto, encoding="utf-8", newline="\n")
    return {nome: hashlib.md5(texto.encode("utf-8")).hexdigest() for nome, texto in arquivos.items()}


def ambiente(**extra):
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", USE_LIBUV="0", **extra)


def rodar_ddp_v0(pasta, args, p, saida):
    argumentos = [sys.executable, "ddp_pinn.py", "--problema", "P1", "--semente", "0", "--saida", saida, *CURTO_CLI, *args]
    if p == 1:
        subprocess.run(argumentos + ["--sequencial"], cwd=pasta, env=ambiente(), check=True)
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        porta = s.getsockname()[1]
    procs = [subprocess.Popen(argumentos, cwd=pasta, env=ambiente(RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE=str(p),
                                                                    MASTER_ADDR="127.0.0.1", MASTER_PORT=str(porta)))
             for r in range(p)]
    assert all(proc.wait() == 0 for proc in procs), "DDP da v0 falhou"


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "v0.1.0-exploratorio"
    commit = git("rev-list", "-n", "1", tag).decode().strip()
    nb = json.loads(git("show", f"{tag}:{CAMINHO_NB}").decode("utf-8"))
    ref = {"tag": tag, "commit": commit, "casos": {}, "ddp": {}, "tarefas": {}}
    with tempfile.TemporaryDirectory() as pasta:
        ref["md5_modulos"] = reconstruir_modulos(nb, pasta)
        saida = os.path.join(pasta, "casos.json")
        subprocess.run([sys.executable, str(Path(__file__).with_name("casos.py")), "v0", pasta, saida],
                       env=ambiente(), check=True)
        ref["casos"] = json.loads(Path(saida).read_text())
        for nome, (args, p) in CASOS_DDP.items():
            saida = os.path.join(pasta, nome + ".json")
            rodar_ddp_v0(pasta, args, p, saida)
            r = json.loads(Path(saida).read_text())
            ref["ddp"][nome] = {"metricas": r["metricas"], "loss": r["hist"]["loss"], "it_final": r["hist"]["it_final"]}
        # A configuracao padrao da tarefa vem do proprio notebook da tag (PADRAO_TAREFA)
        celula = next(c for c in fontes_do_notebook(nb)[0] if "PADRAO_TAREFA = dict(" in c)
        padrao = eval(ast.get_source_segment(celula, next(
            no.value for no in ast.parse(celula).body if isinstance(no, ast.Assign) and no.targets[0].id == "PADRAO_TAREFA")))
        for nome, cfg in CASOS_TAREFA.items():
            saida = os.path.join(pasta, nome + ".json")
            subprocess.run([sys.executable, "tarefa_pinn.py", json.dumps(dict(padrao, **CURTO_TAREFA, **cfg)), saida],
                           cwd=pasta, env=ambiente(), check=True)
            r = json.loads(Path(saida).read_text())
            ref["tarefas"][nome] = {"metricas": r["metricas"], "it_final": r["it_final"], "dtype": r["dtype"]}
    destino = RAIZ / "tests" / "data" / "referencia_v0.json"
    destino.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8")
    print("referencia gravada em", destino, "| modulos:", ref["md5_modulos"])


if __name__ == "__main__":
    main()
