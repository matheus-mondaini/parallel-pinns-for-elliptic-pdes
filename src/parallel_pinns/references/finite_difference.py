"""Referencia FDM de 5 pontos (verificacao independente do pipeline; o erro deve cair como h^2)."""
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


def resolver_fdm(prob, N=63):
    h = 1.0 / (N + 1)
    x = torch.arange(1, N + 1, dtype=torch.float64, device="cpu") * h        # pontos interiores (FDM em CPU)
    X, Y = torch.meshgrid(x, x, indexing="ij")                  # X[i, j] = x_i, Y[i, j] = y_j

    # Matriz 1D de -u'' (sem as linhas de contorno)
    D1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(N, N)) / h**2
    I = sp.identity(N)
    A = (sp.kron(D1, I) + sp.kron(I, D1)).tocsr()               # -lap em 2D

    # Lado direito: fonte + vizinhos de borda (valores de g conhecidos)
    b = prob["f"](X, Y).clone()
    zero = torch.zeros(N, dtype=torch.float64, device="cpu")
    um = torch.ones(N, dtype=torch.float64, device="cpu")
    b[0, :]  += prob["u"](zero, x) / h**2      # borda x = 0
    b[-1, :] += prob["u"](um, x) / h**2        # borda x = 1
    b[:, 0]  += prob["u"](x, zero) / h**2      # borda y = 0
    b[:, -1] += prob["u"](x, um) / h**2        # borda y = 1

    U = spla.spsolve(A, b.numpy().ravel()).reshape(N, N)
    return X.numpy(), Y.numpy(), U
