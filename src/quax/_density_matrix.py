from ._quantum_objects import SuperOp, DensityMatrix
from typing import List, Tuple
from ._apply import targeted_apply_superop


def compute_density_tensor_from_superoperators(
    superoperators: List[SuperOp],
    subsystems: List[Tuple[int, ...]],
    initial_state: DensityMatrix,
) -> DensityMatrix:
    """
    Compute the density matrix from a list of operators, Kraus operators and the subsystems they act on.

    :param superoperators: The superoperators to apply.
    :param subsystems: The qubit indices that the operators act on.
    :param initial_state: The initial state of the system.
    :return: The final density matrix .
    """
    # iterate through the instructions and apply them to the density matrix
    rho = initial_state

    for superop, subsystem in zip(superoperators, subsystems):
        # ρ := Sρ
        rho = targeted_apply_superop(superop, rho, subsystem)

    return rho
