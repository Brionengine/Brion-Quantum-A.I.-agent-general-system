# optimizecircuits.py

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation

def optimize_qiskit_circuit(qc):
    """
    Optimize the Qiskit quantum circuit by reducing gate depth and simplifying operations.
    """
    pass_manager = PassManager([Optimize1qGates(), CommutativeCancellation()])
    optimized_circuit = pass_manager.run(qc)
    return optimized_circuit
