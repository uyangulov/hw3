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


methods = [method1, method2, method3, method4]
method_labels = ["Method 1", "Method 2", "Method 3", "Method 4"]

qubit_range = range(2, 11, 2)
time_values = [1, 5, 10]
p_values = np.logspace(0, 2, 10, base=10, dtype=int)

fig, axes = plt.subplots(len(time_values), len(qubit_range), figsize=(20, 12), sharex=True, sharey=True)

for i, T in enumerate(time_values):
    for j, n in enumerate(qubit_range):
        print(i, j)
        state = psi_init(n)
        omegas, gammas = hamilt_params(n)
        psi_ref = exact_evolution(state, omegas, gammas, T)
        
        fidelities = {label: [] for label in method_labels}
        n_expm_values = {label: [] for label in method_labels}

        for p in p_values:
            dt = T / p
            for method, label in zip(methods, method_labels):
                psi_trotter, n_expm = method(state.copy(), p, omegas, gammas, dt)
                fidelities[label].append(fidelity(psi_ref, psi_trotter))
                n_expm_values[label].append(n_expm)
        
        ax = axes[i, j]
        for label in method_labels:
            ax.plot(n_expm_values[label], 1 - np.array(fidelities[label]), marker='o', label=label)
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(ls=':')
        
        if j == 0:
            ax.set_ylabel(f"T={T}")
        if i == len(time_values) - 1:
            ax.set_xlabel(f"n={n}")

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle("1 - Fidelity", fontsize=16)
plt.tight_layout()
plt.show()