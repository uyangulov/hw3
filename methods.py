import numpy as np
from scipy.sparse.linalg import expm_multiply
from functools import reduce

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

shape_2q_gate = (2, 2, 2, 2)
ZZ = np.kron(Z, Z)
YZ = np.kron(Y, Z)


def dress_with_identity(operator, indices, n):
    """
    Constructs an n-qubit operator with 'operator' applied at 'indices' and identity matrices elsewhere.
    """
    operators = [I2] * n
    for i in indices:
        operators[i] = operator
    return reduce(np.kron, operators)


def hamiltonian_full_matrix(omegas, gammas):
    """
    Constructs the full Hamiltonian matrix for an n-qubit system.

    H = \sum_i \omega_i X_i + \sum_{i,j} \gamma_{i,j} Z_{ij}
    """
    n = len(omegas)
    N = 2**n
    H = np.zeros((N, N), dtype=complex)

    for i in range(n):
        H += omegas[i] * dress_with_identity(X, [i], n)

    for i in range(n):
        for j in range(i):
            ZZ_ij = dress_with_identity(Z, [i, j], n)
            H += gammas[i, j] * ZZ_ij

    return H


def exact_evolution(psi, omegas, gammas, T):
    """
    Computes the exact time evolution of a quantum state.
    Initial state vector must have shape (2, 2, 2, ..., 2) for n qubits.
    """
    H = hamiltonian_full_matrix(omegas, gammas)
    return expm_multiply(-1j*T*H, psi.flatten())


def simple_expm(M, arg):
    """
    Computes the matrix exponential of 1j*arg*M, given that M satisfies M**2 = I.
    """
    I = np.eye(M.shape[0])
    return np.cos(arg) * I + 1j * np.sin(arg) * M


def apply_1q(psi, gate, i):
    """
    Applies a single-qubit gate to the quantum state.
    Initial state vector must have shape (2, 2, 2, ..., 2) for n qubits.
    """
    psi = np.tensordot(gate, psi, axes=[[-1], [i]])
    psi = np.moveaxis(psi, 0, i)
    return psi


def apply_2q(psi, gate, i, j):
    """
    Applies a two-qubit gate to the quantum state.
    Initial state vector must have shape (2, 2, 2, ..., 2) for n qubits.
    """
    psi = np.tensordot(gate.reshape(shape_2q_gate),
                       psi, axes=[[-2, -1], [i, j]])
    psi = np.moveaxis(psi, [0, 1], [i, j])
    return psi


def apply_rx(arg, psi, i):
    """
    Applies an exp(1j * arg * X) rotation to i-th qubit.
    """
    gate = simple_expm(X, arg)
    return apply_1q(psi, gate, i)


def apply_rzz(arg, psi, i, j):
    """
    Applies an exp(1j * arg * Z_i Z_j) transform to qubits i and j.
    """
    gate = simple_expm(ZZ, arg)
    psi = apply_2q(psi, gate, i, j)
    return psi


def apply_ryz(arg, psi, i, j):
    """
    Applies an exp(1j * arg * Y_i Z_j) transform to qubits i and j.
    """
    gate = simple_expm(YZ, arg)
    psi = apply_2q(psi, gate, i, j)
    return psi


def apply_rx_chain(psi, omegas, dt):
    for i, omega in enumerate(omegas):
        psi = apply_rx(-dt * omega, psi, i)
    return psi


def apply_rzz_chain(psi, gammas, dt):
    for i in range(1, len(gammas)):
        for j in range(i):
            psi = apply_rzz(-dt * gammas[i][j], psi, i, j)
    return psi


def apply_ryz_chain(psi, gammas, omegas, dt):
    for i in range(len(gammas)):
        for j in range(len(gammas)):
            if i != j:
                psi = apply_ryz(omegas[i] * gammas[i][j] * dt ** 2, psi, i, j)
    return psi


def method1(psi, p, omegas, gammas, dt):
    for _ in range(p):
        psi = apply_rx_chain(psi, omegas, dt)
        psi = apply_rzz_chain(psi, gammas, dt)

    n = omegas.shape[0]
    # number of matrix exponentials
    # rx + rzz
    n_expm = n + n * (n-1) / 2
    n_expm *= p
    return psi, n_expm


def method2(psi, p, omegas, gammas, dt):
    for _ in range(p):
        psi = apply_rx_chain(psi, omegas, dt)
        psi = apply_rzz_chain(psi, gammas, dt)
        psi = apply_ryz_chain(psi, gammas, omegas, dt)

    n = omegas.shape[0]
    # number of matrix exponentials
    # rx + rzz + ryz
    n_expm = n + n * (n-1) / 2 + n * (n-1)
    n_expm *= p
    return psi, n_expm


def method3(psi, p, omegas, gammas, dt):
    for _ in range(p):
        psi = apply_rx_chain(psi, omegas, dt/2)
        psi = apply_rzz_chain(psi, gammas, dt)
        psi = apply_rx_chain(psi, omegas, dt/2)

    n = omegas.shape[0]
    # number of matrix exponentials
    # rx + rzz + rx
    n_expm = 2 * n + n * (n-1) / 2
    n_expm *= p
    return psi, n_expm


def method4(psi, p, omegas, gammas, dt):
    beta = 1 / (4 - 4**(1/3))
    n_expm = 0

    for _ in range(p):

        for i in range(2):
            psi, n = method3(psi, 1, omegas, gammas, dt * beta)
            n_expm += n

        psi, n = method3(psi, 1, omegas, gammas, dt * (1-4*beta))
        n_expm += n

        for i in range(2):
            psi, n = method3(psi, 1, omegas, gammas, dt * beta)
            n_expm += n

    return psi, n_expm
