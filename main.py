import numpy as np
from methods import exact_evolution, method1, method2, method3, method4
import matplotlib.pyplot as plt


def fidelity(psi1, psi2):
    return np.abs(np.dot(psi1.flatten(), np.conj(psi2.flatten())))**2


def psi_init(n):
    N = 2**n
    state = np.random.randn(N) + 1j * np.random.randn(N)
    state /= np.linalg.norm(state)
    return state.reshape((2,) * n)


def hamilt_params(n):
    omegas = np.random.rand(n)
    gammas = np.random.rand(n, n)
    return omegas, gammas


n = 3
T = 10
p_values = np.logspace(0,4,10,base=10, dtype=int)
state = psi_init(n)
omegas, gammas = hamilt_params(n)

psi_ref = exact_evolution(state, omegas, gammas, T)

methods = [method1, method2, method3, method4]
method_labels = ["Method 1", "Method 2", "Method 3", "Method 4"]

fidelities = {label: [] for label in method_labels}
n_expm_values = {label: [] for label in method_labels}

for p in p_values:
    dt = T / p
    for method, label in zip(methods, method_labels):
        psi_trotter, n_expm = method(state.copy(), p, omegas, gammas, dt)
        fidelities[label].append(fidelity(psi_ref, psi_trotter))
        n_expm_values[label].append(n_expm)

fig, (ax1, ax2) = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)

for label in method_labels:
    ax1.plot(n_expm_values[label], fidelities[label], marker='o', label=label)

ax1.set_ylim(0, 1.1)
ax1.set_xlabel("Number of Matrix Exponentials")
ax1.set_ylabel("Fidelity with Exact Evolution")
ax1.legend()
ax1.set_xscale("log")
ax1.set_title("Fidelity (linear scale)")
ax1.grid(ls=':')

for label in method_labels:
    ax2.plot(n_expm_values[label], 1 - np.array(fidelities[label]), marker='o', label=label)

ax2.set_xlabel("Number of Matrix Exponentials")
ax2.set_ylabel("1 - Fidelity")
ax2.legend()
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_title("1 - Fidelity (log10 scale)")
ax2.grid(ls=':')

# Display the plots
plt.tight_layout()
plt.show()
