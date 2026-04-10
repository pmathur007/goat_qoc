import qutip
import numpy as np
import matplotlib.pyplot as plt

# finite blockade case

ideal_rydberg_basis_names = ["00", "01", "0r", "10", "11", "1r", "r0", "r1", "rr"]
ideal_rydberg_basis = [qutip.basis(9, i) for i in range(9)]
ideal_rydberg_basis_map = {ideal_rydberg_basis_names[i]: ideal_rydberg_basis[i] for i in range(9)}

# general functions for creating an Rydberg Hamiltonians with arbitrary drive strength and phase profile, but fixed detuning
def ideal_rydberg_hamiltonian(drive_strength, phase, extra_parameters=None):
    """drive strength should be real"""
    Omega = qutip.coefficient(lambda t: 0.5 * drive_strength(t) * np.exp(1j * phase(t)))
    Omega_conj = qutip.coefficient(lambda t: 0.5 * drive_strength(t) * np.exp(-1j * phase(t)))

    detuning = extra_parameters["detuning"]
    blockade_strength = extra_parameters["blockade_strength"]

    s = ideal_rydberg_basis_map

    H01 = Omega * (s["01"] * s["0r"].dag()) + Omega_conj * (s["0r"] * s["01"].dag()) - detuning * s["0r"] * s["0r"].dag()
    H10 = Omega * (s["10"] * s["r0"].dag()) + Omega_conj * (s["r0"] * s["10"].dag()) - detuning * s["r0"] * s["r0"].dag()
    H11 = Omega * (s["11"] * s["1r"].dag() + s["11"] * s["r1"].dag() + s["1r"] * s["rr"].dag() + s["r1"] * s["rr"].dag()) \
        + Omega_conj * (s["1r"] * s["11"].dag() + s["r1"] * s["11"].dag() + s["rr"] * s["1r"].dag() + s["rr"] * s["r1"].dag()) \
        - detuning * (s["1r"] * s["1r"].dag() + s["r1"] * s["r1"].dag()) \
        + (blockade_strength - 2*detuning) * s["rr"] * s["rr"].dag()

    return H01 + H10 + H11

# infinite blockade case

reduced_ideal_rydberg_basis_names = ["01", "11", "0r", "W"]
reduced_ideal_rydberg_basis = [qutip.basis(4, i) for i in range(4)]
reduced_ideal_rydberg_basis_map = {reduced_ideal_rydberg_basis_names[i]: reduced_ideal_rydberg_basis[i] for i in range(4)}

def reduced_ideal_rydberg_hamiltonian(drive_strength, phase, extra_parameters=None):
    """drive strength should be real"""
    Omega = qutip.coefficient(lambda t: 0.5 * drive_strength(t) * np.exp(1j * phase(t)))
    Omega_conj = qutip.coefficient(lambda t: 0.5 * drive_strength(t) * np.exp(-1j * phase(t)))

    Omega_enhanced = qutip.coefficient(lambda t: 0.5 * np.sqrt(2) * drive_strength(t) * np.exp(1j * phase(t)))
    Omega_conj_enhanced = qutip.coefficient(lambda t: 0.5 * np.sqrt(2) * drive_strength(t) * np.exp(-1j * phase(t)))

    s = reduced_ideal_rydberg_basis_map

    H01 = Omega * (s["01"] * s["0r"].dag()) + Omega_conj * (s["0r"] * s["01"].dag())
    H11 = Omega_enhanced * (s["11"] * s["W"].dag()) + Omega_conj_enhanced * (s["W"] * s["11"].dag())
    if extra_parameters is not None:
        detuning = extra_parameters["detuning"]
        H01 = H01 - detuning * (s["0r"] * s["0r"].dag())
        H11 = H11 - detuning * (s["W"] * s["W"].dag())

    return H01 + H11

# utility function for getting pulse parameters at arbitrary times
def data_from_tstep(t, t_steps, data):
    i = 0
    while i < len(t_steps)-1 and t_steps[i+1] < t:
        i += 1
    return data[i]

# utility function for plotting
def plot_populations(results, psi0_is, psi0_strs, ts):
    for i in range(len(psi0_is)):
        P0 = np.zeros(ts.size)
        for t in range(ts.size):
            P0[t] = np.abs(results[i].states[t].full()[psi0_is[i], 0]) ** 2 # IMPORTANT: this assumes that the psi_strs are in basis order!!!! TODO: fix this
        plt.plot(ts, P0, label=f"{psi0_strs[i]}")
    plt.ylim((-0.05, 1.05))
    plt.xlabel("$t\\Omega_{\\text{max}}$") # , fontsize=16)
    plt.ylabel("Population") # , fontsize=16)
    plt.legend(title="State")
    plt.grid()
    # plt.show()

def simulate_gate(hamiltonian_name, drive_strength_func, phase_func, t_min, t_max, nts_sim, extra_parameters=None, plot=True):
    ts = np.linspace(t_min, t_max, nts_sim)

    if hamiltonian_name == "ideal_rydberg":
        psi0_is = [0, 1, 3, 4]
        psi0_strs = [ideal_rydberg_basis_names[psi0_i] for psi0_i in psi0_is]
        psi0s = [ideal_rydberg_basis[psi0_i] for psi0_i in psi0_is] 
        hamiltonian_func = ideal_rydberg_hamiltonian
    else: # hamiltonian_name == "reduced_ideal_rydberg"
        psi0_is = [0, 1]
        psi0_strs = [reduced_ideal_rydberg_basis_names[psi0_i] for psi0_i in psi0_is]
        psi0s = [reduced_ideal_rydberg_basis[psi0_i] for psi0_i in psi0_is] 
        hamiltonian_func = reduced_ideal_rydberg_hamiltonian

    H = hamiltonian_func(drive_strength_func,
                         phase_func,
                         extra_parameters)
    
    options = {
        "max_step": 0.01,
        "atol": 1e-12,
        "rtol": 1e-10,
        "method": "dop853"
    }

    results = [qutip.sesolve(H, psi0, ts, e_ops=None, options=options) for psi0 in psi0s]
    
    if plot:
        plot_populations(results, psi0_is, psi0_strs, ts)

    return results

def calculate_fidelity(hamiltonian_name, metric, results, single_qubit_phase = 0):
    if hamiltonian_name == "ideal_rydberg":
        target_states = [
            qutip.Qobj([1, 0, 0, 0]),
            qutip.Qobj([0, 1, 0, 0]),
            qutip.Qobj([0, 0, 1, 0]),
            qutip.Qobj([0, 0, 0, -1])
        ]

        sq_gate = qutip.Qobj([[1, 0, 0, 0],
                              [0, np.exp(1j * single_qubit_phase), 0, 0],
                              [0, 0, np.exp(1j * single_qubit_phase), 0],
                              [0, 0, 0, np.exp(2j * single_qubit_phase)]])
        a00 = target_states[0].dag() * sq_gate * qutip.Qobj(results[0].states[-1].full().flatten()[[0, 1, 3, 4]])
        a01 = target_states[1].dag() * sq_gate * qutip.Qobj(results[1].states[-1].full().flatten()[[0, 1, 3, 4]])
        a10 = target_states[2].dag() * sq_gate * qutip.Qobj(results[2].states[-1].full().flatten()[[0, 1, 3, 4]])
        a11 = target_states[3].dag() * sq_gate * qutip.Qobj(results[3].states[-1].full().flatten()[[0, 1, 3, 4]])

        if metric == "TO":
            return (1/20) * (np.abs(a00 + a01 + a10 + a11)**2 
                             + np.abs(a00)**2 + np.abs(a01)**2 + np.abs(a10)**2 + np.abs(a11)**2)
        else: # metric == "GOAT"
            return (1/4) * np.abs(a00 + a01 + a10 + a11)
    else: # hamiltonian_name == "reduced_ideal_rydberg"
        target_states = [
            qutip.Qobj([1, 0, 0, 0]),
            qutip.Qobj([0, -1, 0, 0]),
        ]

        sq_gate = qutip.Qobj([[np.exp(1j * single_qubit_phase), 0, 0, 0],
                              [0, np.exp(2j * single_qubit_phase), 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])

        a01 = target_states[0].dag() * sq_gate * qutip.Qobj(results[0].states[-1].full())
        a11 = target_states[1].dag() * sq_gate * qutip.Qobj(results[1].states[-1].full())

        if metric == "TO":
            return (1/20) * (np.abs(1 + 2*a01 + a11)**2 + (1 + 2*np.abs(a01)**2 + np.abs(a11)**2))
        else: # metric == "GOAT"
            return (1/4) * np.abs(1 + 2*a01 + a11)