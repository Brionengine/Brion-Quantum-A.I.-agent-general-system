# optimizecircuits.py
from __future__ import annotations


try:
    from qiskit.transpiler import PassManager
except ImportError:  # optional dependency: pip install qiskit
    PassManager = None
try:
    from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
except ImportError:  # optional dependency: pip install qiskit
    Optimize1qGates = None
    CommutativeCancellation = None

def optimize_qiskit_circuit(qc):
    """
    Optimize the Qiskit quantum circuit by reducing gate depth and simplifying operations.
    """
    pass_manager = PassManager([Optimize1qGates(), CommutativeCancellation()])
    optimized_circuit = pass_manager.run(qc)
    return optimized_circuit
