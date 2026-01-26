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

import jax.numpy as jnp

from ._quantum_objects import DensityMatrix, StateVector


def promote_state_vector_to_density_matrix(
    state: StateVector,
) -> DensityMatrix:
    """
    Promote a state vector to a density matrix.

    :param state: State vector to promote.
    :return: Density matrix corresponding to the state vector.
    """
    # Use matrix form to compute outer product: ρ = |ψ⟩⟨ψ|
    state_vec = state.matrix  # shape (*ensemble, d)
    rho_matrix = jnp.einsum("...a,...b->...ab", state_vec, jnp.conj(state_vec))
    return DensityMatrix.from_matrix(rho_matrix, state.dims, state.num_ensemble_dims)
