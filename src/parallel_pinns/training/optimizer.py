"""Laco de otimizacao comum a todas as arquiteturas: Adam (epocas fixas) e depois L-BFGS (ate o plato)."""
import math
import time

import torch


def otimizar(parametros, perdas, registrar, mostrar, relogio, media_global, epocas_adam, passos_lbfgs_max,
             iter_por_passo, lr, tolerancia_plato, janela_plato, avaliar_a_cada, somar_gradientes=lambda: None):
    # Adam com numero fixo de epocas, depois L-BFGS ate o plato (ou ate passos_lbfgs_max).
    # Plato: a perda caiu menos que tolerancia_plato (relativo) nos ultimos janela_plato passos.
    # somar_gradientes: chamado depois de cada backward (XPINN distribuido: all-reduce dos gradientes).
    # Na GPU as operacoes sao assincronas: sincroniza antes de ler o relogio
    sincronizar = torch.cuda.synchronize if parametros[0].is_cuda else (lambda: None)
    adam = torch.optim.Adam(parametros, lr=lr)
    for epoca in range(epocas_adam):
        t0 = time.perf_counter()
        adam.zero_grad()
        termos = perdas()
        termos[0].backward()                              # no DDP, aqui acontece o all-reduce
        somar_gradientes()
        adam.step()
        sincronizar()
        relogio[0] += time.perf_counter() - t0            # so o treino entra no relogio
        valores = [media_global(v).item() for v in termos]
        m = registrar(epoca, valores, forcar=(epoca % avaliar_a_cada == 0))
        if epoca % 1000 == 0:
            mostrar("Adam ", epoca, valores, m)

    lbfgs = torch.optim.LBFGS(parametros, lr=1.0, max_iter=iter_por_passo, history_size=50,
                              line_search_fn="strong_wolfe", tolerance_grad=0, tolerance_change=0)
    def closure():
        lbfgs.zero_grad()
        loss = perdas()[0]
        loss.backward()
        somar_gradientes()
        # A busca de linha compara valores da perda: todos os processos precisam ver o MESMO valor
        return media_global(loss)

    it, perda_passos, passos = epocas_adam, [], 0
    for passo in range(passos_lbfgs_max):
        t0 = time.perf_counter()
        perda_passo = float(lbfgs.step(closure).detach())
        sincronizar()
        relogio[0] += time.perf_counter() - t0
        it += iter_por_passo
        passos += 1
        perda_passos.append(perda_passo)
        avaliar_agora = (passo + 1) % (avaliar_a_cada // iter_por_passo) == 0
        valores = [media_global(v).item() for v in perdas(registro=True)]
        m = registrar(it, valores, forcar=avaliar_agora)
        if (passo + 1) % 25 == 0:
            mostrar("LBFGS", it, valores, m)
        if not math.isfinite(perda_passo):
            break
        if len(perda_passos) > janela_plato:
            antes = perda_passos[-janela_plato - 1]
            if (antes - perda_passo) / abs(antes) < tolerancia_plato:
                break
    registrar(it, valores, forcar=True)                   # avaliacao final sempre registrada
    return it, passos
