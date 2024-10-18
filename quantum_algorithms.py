# quantum_algorithms.py

from qiskit import QuantumCircuit
import cirq

def generate_qiskit_circuit(n_qubits):
    """
    Generate a simple Qiskit quantum circuit based on the problem.
    """
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))  # Apply Hadamard gates to all qubits
    qc.measure_all()
    return qc

def generate_cirq_circuit(n_qubits):
    """
    Generate a simple Cirq quantum circuit based on the problem.
    """
    qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
    circuit = cirq.Circuit()
    circuit.append([cirq.H(q) for q in qubits])  # Apply Hadamard gates
    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit
