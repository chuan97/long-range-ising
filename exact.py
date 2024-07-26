import math
from collections import namedtuple

import numpy as np
from scipy.linalg import expm, kron
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse import kron as krons
from scipy.sparse.linalg import expm as expms


def antiferro_hom_unfrustrated_small_spins(wx, wz, G, N, s):
    Sz, Sp, Sm, Seye = spin_operators(s)
    Sx = 0.5 * (Sp + Sm)

    # ising interaction
    aux_op_A = csr_matrix((int(2 * s + 1) ** N, int(2 * s + 1) ** N))
    aux_op_B = csr_matrix((int(2 * s + 1) ** N, int(2 * s + 1) ** N))
    for i in range(N):
        if i % 2 == 0:
            op_chain_A = [Seye] * i + [Sx] + [Seye] * (N - i - 1)
            aux_op_A += sparse_kron(*op_chain_A)
        if i % 2 == 1:
            op_chain_B = [Seye] * i + [Sx] + [Seye] * (N - i - 1)
            aux_op_B += sparse_kron(*op_chain_B)
    Hint = -G / N * (aux_op_A - aux_op_B) @ (aux_op_A - aux_op_B)

    # transverse field
    Hz = csr_matrix((int(2 * s + 1) ** N, int(2 * s + 1) ** N))
    for i in range(N):
        op_chain = [Seye] * i + [Sz] + [Seye] * (N - i - 1)
        Hz += -wz * sparse_kron(*op_chain)

    # longitudinal field
    Hx = csr_matrix((int(2 * s + 1) ** N, int(2 * s + 1) ** N))
    for i in range(N):
        op_chain = [Seye] * i + [Sx] + [Seye] * (N - i - 1)
        Hx += -wx * sparse_kron(*op_chain)

    return Hz + Hx + Hint


def pureantiferro_hom_unfrustrated_small_spins(wx, wz, G, N, s):
    Sz, Sp, Sm, Seye = spin_operators(s)
    Sx = 0.5 * (Sp + Sm)

    # ising interaction
    aux_op_A = csr_matrix((int(2 * s + 1) ** N, int(2 * s + 1) ** N))
    aux_op_B = csr_matrix((int(2 * s + 1) ** N, int(2 * s + 1) ** N))
    for i in range(N):
        if i % 2 == 0:
            op_chain_A = [Seye] * i + [Sx] + [Seye] * (N - i - 1)
            aux_op_A += sparse_kron(*op_chain_A)
        if i % 2 == 1:
            op_chain_B = [Seye] * i + [Sx] + [Seye] * (N - i - 1)
            aux_op_B += sparse_kron(*op_chain_B)
    Hint = 4 * G / N * aux_op_A @ aux_op_B

    # transverse field
    Hz = csr_matrix((int(2 * s + 1) ** N, int(2 * s + 1) ** N))
    for i in range(N):
        op_chain = [Seye] * i + [Sz] + [Seye] * (N - i - 1)
        Hz += -wz * sparse_kron(*op_chain)

    # longitudinal field
    Hx = csr_matrix((int(2 * s + 1) ** N, int(2 * s + 1) ** N))
    for i in range(N):
        op_chain = [Seye] * i + [Sx] + [Seye] * (N - i - 1)
        Hx += -wx * sparse_kron(*op_chain)

    return Hz + Hx + Hint


def antiferro_hom_unfrustrated_big_spins(wx, wz, G, N, s):
    S = s * N / 2
    Sz, Sp, Sm, Seye = spin_operators(S)
    Sx = 0.5 * (Sp + Sm)

    SxA = sparse_kron(Sx, Seye)
    SxB = sparse_kron(Seye, Sx)
    SzA = sparse_kron(Sz, Seye)
    SzB = sparse_kron(Seye, Sz)

    # ising interaction
    Hint = -G / N * (SxA - SxB) @ (SxA - SxB)

    # transverse field
    Hz = -wz * (SzA + SzB)

    # longitudinal field
    Hx = -wx * (SxA + SxB)

    return Hz + Hx + Hint


def pureantiferro_hom_unfrustrated_big_spins(wx, wz, G, N, s):
    S = s * N / 2
    Sz, Sp, Sm, Seye = spin_operators(S)
    Sx = 0.5 * (Sp + Sm)

    SxA = sparse_kron(Sx, Seye)
    SxB = sparse_kron(Seye, Sx)
    SzA = sparse_kron(Sz, Seye)
    SzB = sparse_kron(Seye, Sz)

    # ising interaction
    Hint = 4 * G / N * SxA @ SxB

    # transverse field
    Hz = -wz * (SzA + SzB)

    # longitudinal field
    Hx = -wx * (SxA + SxB)

    return Hz + Hx + Hint


def spin_operators(S, *, to_dense_array=False, format=None, dtype=np.float_):
    Sz = diags([m for m in np.arange(-S, S + 1)], format=format, dtype=dtype)
    Sp = diags(
        [math.sqrt(S * (S + 1) - m * (m + 1)) for m in np.arange(-S, S)],
        offsets=-1,
        format=format,
        dtype=dtype,
    )
    Sm = Sp.T
    Seye = eye(2 * S + 1, format=format, dtype=dtype)

    Spin_operators = namedtuple("Spin_operators", "Sz Sp Sm Seye")
    ops = Spin_operators(Sz, Sp, Sm, Seye)
    if to_dense_array:
        ops = Spin_operators(*[o.toarray() for o in ops])

    return ops


def sparse_kron_factory(n):
    """
    function factory: returns a function that computes the kronecker product of n sparse matrices
    """

    import scipy.sparse as sp

    if n == 2:
        return sp.kron
    else:

        def inner(*ops, format=None):
            if len(ops) != n:
                raise TypeError(
                    f"sparse_kron_factory({n})() takes exactly {n} arguments"
                )

            return sp.kron(ops[0], sparse_kron_factory(n - 1)(*ops[1:]), format=format)

        inner.__name__ = f"sparse_kron_factory({n})"
        inner.__module__ = __name__
        inner.__doc__ = (
            f"kronecker product of sparse matrices A1, ..., An with n={n}\n\n"
        )

        return inner


def sparse_kron(*ops, format=None):
    """
    kronecker product of an arbitrary number of sparse matrices
    """
    if len(ops) < 2:
        raise TypeError("sparse_kron takes at least two arguments")

    return sparse_kron_factory(len(ops))(*ops, format=format)


def sort_eigensystem(vals, vects=None):
    if vects is None:
        return np.sort(vals)
    else:
        idx = np.argsort(vals)
        return vals[idx], vects[:, idx]


from scipy.sparse import spmatrix


def lanczos_ed(
    operator: spmatrix,
    *,
    k: int = 1,
    compute_eigenvectors: bool = False,
    scipy_args: dict = None,
):
    r"""
    *** Adapted from Netket ***

    Computes `first_n` smallest eigenvalues and, optionally, eigenvectors
    of a Hermitian operator using :meth:`scipy.sparse.linalg.eigsh`.

    Args:
        operator: Scipy sparse matrix to diagonalize.
        k: The number of eigenvalues to compute.
        compute_eigenvectors: Whether or not to return the
            eigenvectors of the operator. With ARPACK, not requiring the
            eigenvectors has almost no performance benefits.
        scipy_args: Additional keyword arguments passed to
            :meth:`scipy.sparse.linalg.eigvalsh`. See the Scipy documentation for further
            information.

    Returns:
        Either `w` or the tuple `(w, v)` depending on whether `compute_eigenvectors`
        is True.

        - w: Array containing the lowest `first_n` eigenvalues.
        - v: Array containing the eigenvectors as columns, such that`v[:, i]`
          corresponds to `w[i]`.
    """
    from scipy.sparse.linalg import eigsh

    actual_scipy_args = {}
    if scipy_args:
        actual_scipy_args.update(scipy_args)
    actual_scipy_args["which"] = "SA"
    actual_scipy_args["k"] = k
    actual_scipy_args["return_eigenvectors"] = compute_eigenvectors

    result = eigsh(operator, **actual_scipy_args)
    if not compute_eigenvectors:
        # for some reason scipy does a terrible job
        # ordering the eigenvalues and eigenvectors
        # therefore we do it ourselves
        return sort_eigensystem(result)
    else:
        return sort_eigensystem(*result)
