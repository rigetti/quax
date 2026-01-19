# Copyright 2026 Rigetti & Co, LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX-based quantum operator transformations."""

from ._apply import (
    apply_choi_to_density_matrix,
    apply_kraus_to_density_matrix,
    apply_kraus_to_state_vector,
    apply_pauli_liouville_to_density_matrix,
    apply_superop_to_density_matrix,
    apply_unitary_to_state_vector,
    compute_choi_observables_from_states,
    compute_kraus_observables_from_states,
    compute_pauli_liouville_observables_from_states,
    compute_superop_observables_from_states,
    partial_trace,
)
from ._common_channels import (
    depolarizing_channel_superoperator,
    integrated_thermal_superoperator,
    thermal_relaxation_choi,
    bit_flip_operators,
    phase_flip_operators,
    bitphase_flip_operators,
    dephasing_operators,
    depolarizing_operators,
    relaxation_operators,
    KRAUS_OPS,
)
from ._compose import (
    compose_choi,
    compose_kraus,
    compose_pauli_liouville,
    compose_superop,
    compose_unitary,
)
from ._distance_metrics import (
    average_fidelity_to_depolarizing_constant,
    average_fidelity_to_process_fidelity,
    depolarizing_constant_to_average_fidelity,
    depolarizing_constant_to_process_fidelity,
    fidelity,
    process_fidelity,
    process_fidelity_to_average_fidelity,
    process_fidelity_to_depolarizing_constant,
    unitarity_to_stochastic_infidelity,
    unitary_entanglement_fidelity,
)
from ._power import power_choi, power_kraus, power_pauli_liouville, power_superop, power_unitary
from ._promotion import promote_state_vector_to_density_matrix
from ._quantum_objects import (
    Choi,
    DensityMatrix,
    Kraus,
    KrausMap,
    Operator,
    PauliLiouville,
    State,
    StateVector,
    SuperOp,
    SuperOperator,
    Unitary,
)
from ._random import (
    ginibre_matrix_complex,
    random_choi_BCSZ,
    random_density_matrix,
    random_state_vector,
    random_unitary,
)
from ._state import (
    mixed_state_matrix,
    tensor_density_matrices,
    tensor_state_vectors,
    zero_state_matrix,
    zero_state_vector,
)
from ._superoperator_transformations import (
    choi_to_kraus,
    choi_to_pauli_liouville,
    choi_to_superop,
    kraus_to_choi,
    kraus_to_pauli_liouville,
    kraus_to_superop,
    pauli_liouville_to_choi,
    pauli_liouville_to_kraus,
    pauli_liouville_to_superop,
    superop_to_choi,
    superop_to_kraus,
    superop_to_pauli_liouville,
    to_choi,
    to_kraus,
    to_pauli_liouville,
    to_superop,
    unitary_to_choi,
    unitary_to_pauli_liouville,
    unitary_to_superop,
)
from ._tensor import (
    tensor_choi,
    tensor_density_matrix,
    tensor_kraus,
    tensor_pauli_liouville,
    tensor_state_vector,
    tensor_superop,
    tensor_unitary,
)
from ._validation import (
    is_completely_positive,
    is_cptp,
    is_identity_matrix,
    is_one_design,
    is_trace_preserving,
    is_two_design,
    is_unitary,
)

# Import gates, states and ensembles as submodules
from . import gates
from . import states
from . import ensembles

__all__ = [
    # Apply superoperator functions
    "compute_choi_observables_from_states",
    "compute_kraus_observables_from_states",
    "compute_pauli_liouville_observables_from_states",
    "compute_superop_observables_from_states",
    "partial_trace",
    "apply_choi_to_density_matrix",
    "apply_kraus_to_density_matrix",
    "apply_pauli_liouville_to_density_matrix",
    "apply_superop_to_density_matrix",
    "apply_unitary_to_state_vector",
    "apply_kraus_to_state_vector",
    # Common channels
    "depolarizing_channel_superoperator",
    "integrated_thermal_superoperator",
    "thermal_relaxation_choi",
    "bit_flip_operators",
    "phase_flip_operators",
    "bitphase_flip_operators",
    "dephasing_operators",
    "depolarizing_operators",
    "relaxation_operators",
    "KRAUS_OPS",
    # Compose quantum objects
    "compose_kraus",
    "compose_choi",
    "compose_pauli_liouville",
    "compose_superop",
    "compose_unitary",
    # Tensor quantum objects
    "tensor_kraus",
    "tensor_choi",
    "tensor_pauli_liouville",
    "tensor_superop",
    "tensor_unitary",
    "tensor_state_vector",
    "tensor_density_matrix",
    # Superoperator transformations
    "choi_to_kraus",
    "choi_to_pauli_liouville",
    "choi_to_superop",
    "kraus_to_choi",
    "kraus_to_pauli_liouville",
    "kraus_to_superop",
    "pauli_liouville_to_choi",
    "pauli_liouville_to_kraus",
    "pauli_liouville_to_superop",
    "superop_to_choi",
    "superop_to_kraus",
    "superop_to_pauli_liouville",
    "to_choi",
    "to_pauli_liouville",
    "to_superop",
    "to_kraus",
    "unitary_to_choi",
    "unitary_to_pauli_liouville",
    "unitary_to_superop",
    # Distance metrics
    "fidelity",
    "process_fidelity",
    "to_choi",
    "unitary_entanglement_fidelity",
    "depolarizing_constant_to_average_fidelity",
    "average_fidelity_to_depolarizing_constant",
    "depolarizing_constant_to_process_fidelity",
    "process_fidelity_to_depolarizing_constant",
    "average_fidelity_to_process_fidelity",
    "process_fidelity_to_average_fidelity",
    "unitarity_to_stochastic_infidelity",
    # Types
    "Choi",
    "Kraus",
    "SuperOp",
    "KrausMap",
    "PauliLiouville",
    "SuperOperator",
    "Unitary",
    "DensityMatrix",
    "StateVector",
    "State",
    "Operator",
    # State functions
    "zero_state_matrix",
    "zero_state_vector",
    "mixed_state_matrix",
    "tensor_density_matrices",
    "tensor_state_vectors",
    # Random functions
    "ginibre_matrix_complex",
    "random_density_matrix",
    "random_unitary",
    "random_choi_BCSZ",
    "random_state_vector",
    # Validation functions
    "is_unitary",
    "is_one_design",
    "is_two_design",
    "is_identity_matrix",
    "is_cptp",
    "is_completely_positive",
    "is_trace_preserving",
    # Promotion functions
    "promote_state_vector_to_density_matrix",
    # Power functions
    "power_choi",
    "power_kraus",
    "power_pauli_liouville",
    "power_superop",
    "power_unitary",
    # Submodules
    "gates",
    "states",
    "ensembles",
]
