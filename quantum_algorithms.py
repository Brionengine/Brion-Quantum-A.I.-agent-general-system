# quantum_algorithms.py
# Brion Quantum - Advanced Quantum Algorithm Library v2.0
# Supports: Grover's Search, VQE, QAOA, QFT, Quantum Entanglement, Bell States

import numpy as np
from qiskit import QuantumCircuit
import cirq


# ============================================================================
# Basic Circuit Generators
# ============================================================================

def generate_qiskit_circuit(n_qubits):
    """Generate a simple Qiskit quantum circuit with superposition."""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.measure_all()
    return qc


def generate_cirq_circuit(n_qubits):
    """Generate a simple Cirq quantum circuit with superposition."""
    qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
    circuit = cirq.Circuit()
    circuit.append([cirq.H(q) for q in qubits])
    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit


# ============================================================================
# Advanced Quantum Algorithms
# ============================================================================

def generate_bell_state(backend='qiskit'):
    """
    Generate a maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2.
    Foundation for quantum teleportation and superdense coding.
    """
    if backend == 'qiskit':
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        return qc
    else:
        qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)]
        circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.measure(*qubits, key='result')
        ])
        return circuit


def generate_ghz_state(n_qubits, backend='qiskit'):
    """
    Generate a GHZ (Greenberger-Horne-Zeilinger) state.
    |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
    Maximally entangled state useful for quantum error correction.
    """
    if backend == 'qiskit':
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n_qubits), range(n_qubits))
        return qc
    else:
        qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
        ops = [cirq.H(qubits[0])]
        for i in range(n_qubits - 1):
            ops.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        ops.append(cirq.measure(*qubits, key='result'))
        return cirq.Circuit(ops)


def generate_qft_circuit(n_qubits, backend='qiskit'):
    """
    Quantum Fourier Transform - the quantum analog of the discrete Fourier transform.
    Core subroutine of Shor's algorithm and quantum phase estimation.
    Complexity: O(n^2) gates for n qubits.
    """
    if backend == 'qiskit':
        qc = QuantumCircuit(n_qubits, n_qubits)
        for i in range(n_qubits):
            qc.h(i)
            for j in range(i + 1, n_qubits):
                angle = np.pi / (2 ** (j - i))
                qc.cp(angle, j, i)
        # Swap qubits for correct ordering
        for i in range(n_qubits // 2):
            qc.swap(i, n_qubits - i - 1)
        qc.measure(range(n_qubits), range(n_qubits))
        return qc
    else:
        qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
        ops = []
        for i in range(n_qubits):
            ops.append(cirq.H(qubits[i]))
            for j in range(i + 1, n_qubits):
                angle = np.pi / (2 ** (j - i))
                ops.append(cirq.CZPowGate(exponent=angle / np.pi)(qubits[j], qubits[i]))
        for i in range(n_qubits // 2):
            ops.append(cirq.SWAP(qubits[i], qubits[n_qubits - i - 1]))
        ops.append(cirq.measure(*qubits, key='result'))
        return cirq.Circuit(ops)


def generate_variational_circuit(n_qubits, params, depth=2, backend='qiskit'):
    """
    Parameterized variational circuit (ansatz) for VQE and QAOA.
    Uses alternating layers of rotations and entangling gates.
    BrionQuantum Variational Ansatz (BQVA) design.

    Args:
        n_qubits: Number of qubits
        params: Flat array of rotation angles, length = n_qubits * depth * 3
        depth: Number of variational layers
        backend: 'qiskit' or 'cirq'
    """
    param_idx = 0
    if backend == 'qiskit':
        qc = QuantumCircuit(n_qubits, n_qubits)
        for layer in range(depth):
            # Rotation layer: Rx, Ry, Rz on each qubit
            for q in range(n_qubits):
                if param_idx < len(params):
                    qc.rx(params[param_idx], q)
                    param_idx += 1
                if param_idx < len(params):
                    qc.ry(params[param_idx], q)
                    param_idx += 1
                if param_idx < len(params):
                    qc.rz(params[param_idx], q)
                    param_idx += 1
            # Entangling layer: CNOT chain
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            if n_qubits > 2:
                qc.cx(n_qubits - 1, 0)  # Circular entanglement
        qc.measure(range(n_qubits), range(n_qubits))
        return qc
    else:
        qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
        ops = []
        for layer in range(depth):
            for q in range(n_qubits):
                if param_idx < len(params):
                    ops.append(cirq.rx(params[param_idx])(qubits[q]))
                    param_idx += 1
                if param_idx < len(params):
                    ops.append(cirq.ry(params[param_idx])(qubits[q]))
                    param_idx += 1
                if param_idx < len(params):
                    ops.append(cirq.rz(params[param_idx])(qubits[q]))
                    param_idx += 1
            for q in range(n_qubits - 1):
                ops.append(cirq.CNOT(qubits[q], qubits[q + 1]))
            if n_qubits > 2:
                ops.append(cirq.CNOT(qubits[n_qubits - 1], qubits[0]))
        ops.append(cirq.measure(*qubits, key='result'))
        return cirq.Circuit(ops)


def generate_grover_oracle(n_qubits, target_state, backend='qiskit'):
    """
    Generate Grover's oracle for a specific target state.
    Marks the target state with a phase flip.

    Args:
        n_qubits: Number of qubits
        target_state: Integer representing target (e.g., 5 for |101⟩)
        backend: 'qiskit' or 'cirq'
    """
    if backend == 'qiskit':
        qc = QuantumCircuit(n_qubits)
        # Flip qubits where target has 0
        target_bits = format(target_state, f'0{n_qubits}b')
        for i, bit in enumerate(reversed(target_bits)):
            if bit == '0':
                qc.x(i)
        # Multi-controlled Z gate
        if n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        # Unflip
        for i, bit in enumerate(reversed(target_bits)):
            if bit == '0':
                qc.x(i)
        return qc
    else:
        qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
        ops = []
        target_bits = format(target_state, f'0{n_qubits}b')
        for i, bit in enumerate(reversed(target_bits)):
            if bit == '0':
                ops.append(cirq.X(qubits[i]))
        if n_qubits >= 2:
            ops.append(cirq.Z(qubits[-1]).controlled_by(*qubits[:-1]))
        for i, bit in enumerate(reversed(target_bits)):
            if bit == '0':
                ops.append(cirq.X(qubits[i]))
        return cirq.Circuit(ops)


def generate_grover_diffusion(n_qubits, backend='qiskit'):
    """
    Generate Grover's diffusion operator (inversion about average).
    """
    if backend == 'qiskit':
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
        return qc
    else:
        qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
        ops = []
        ops.extend([cirq.H(q) for q in qubits])
        ops.extend([cirq.X(q) for q in qubits])
        if n_qubits >= 2:
            ops.append(cirq.Z(qubits[-1]).controlled_by(*qubits[:-1]))
        ops.extend([cirq.X(q) for q in qubits])
        ops.extend([cirq.H(q) for q in qubits])
        return cirq.Circuit(ops)


def generate_full_grover(n_qubits, target_state, backend='qiskit'):
    """
    Complete Grover's search algorithm with optimal iteration count.
    Achieves quadratic speedup: O(√N) vs O(N) classical search.

    Args:
        n_qubits: Number of qubits (searches 2^n items)
        target_state: Integer target to find
        backend: 'qiskit' or 'cirq'
    """
    num_iterations = max(1, int(np.pi / 4 * np.sqrt(2 ** n_qubits)))

    if backend == 'qiskit':
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(range(n_qubits))
        oracle = generate_grover_oracle(n_qubits, target_state, 'qiskit')
        diffusion = generate_grover_diffusion(n_qubits, 'qiskit')
        for _ in range(num_iterations):
            qc.compose(oracle, inplace=True)
            qc.compose(diffusion, inplace=True)
        qc.measure(range(n_qubits), range(n_qubits))
        return qc
    else:
        qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
        oracle = generate_grover_oracle(n_qubits, target_state, 'cirq')
        diffusion = generate_grover_diffusion(n_qubits, 'cirq')
        circuit = cirq.Circuit([cirq.H(q) for q in qubits])
        for _ in range(num_iterations):
            circuit += oracle
            circuit += diffusion
        circuit.append(cirq.measure(*qubits, key='result'))
        return circuit


def estimate_circuit_resources(n_qubits, depth, gate_count):
    """
    Estimate computational resources needed for a quantum circuit.

    Returns:
        Dict with memory, time, and fidelity estimates
    """
    statevector_memory_bytes = (2 ** n_qubits) * 16  # Complex128
    estimated_time_ms = gate_count * 0.05  # ~50μs per gate
    estimated_fidelity = (1 - 0.001) ** gate_count  # 0.1% error per gate

    return {
        'qubits': n_qubits,
        'depth': depth,
        'gate_count': gate_count,
        'statevector_memory_mb': statevector_memory_bytes / (1024 * 1024),
        'estimated_time_ms': estimated_time_ms,
        'estimated_fidelity': estimated_fidelity,
        'feasible': n_qubits <= 30 and estimated_fidelity > 0.5,
    }
