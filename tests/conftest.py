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

"""Test fixtures for JAX operator tools tests."""

import jax
import jax.numpy as jnp
import pytest
import qutip as qt

from quax import (
    Choi,
    KrausMap,
    PauliLiouville,
    SuperOp,
    random_choi_BCSZ,
    random_unitary,
)

from .reference_pauli_liouville import choi2pauli_liouville, kraus2pauli_liouville


@pytest.fixture(scope="session", params=[58, 3854])
def seed(request):
    """Parametrized seed fixture for reproducible randomness."""
    return request.param


@pytest.fixture(scope="session", params=[1, 2, 3])
def num_qubits(request):
    """Parametrized number of qubits fixture."""
    return request.param


@pytest.fixture(scope="session", params=[(), (3,), (3, 4)])
def ensemble_size(request):
    """Parametrized ensemble size fixture."""
    return request.param


@pytest.fixture(scope="session")
def random_choi_channels(seed, num_qubits, ensemble_size):
    """
    Generate random quantum channel as Choi matrix using BCSZ distribution.

    Session-scoped fixture parametrized over seeds [58, 3854]
    and num_qubits [1, 2, 3], providing 12 different random channels.

    :param seed: Random seed for JAX PRNG
    :param num_qubits: Number of qubits for the channel
    :return: Choi object representing a random CPTP channel
    """
    rank = 2**num_qubits
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    key = jax.random.PRNGKey(seed)
    return random_choi_BCSZ(dims=dims, rank=rank, key=key, size=ensemble_size)


@pytest.fixture(scope="session")
def random_unitaries(seed, num_qubits, ensemble_size):
    """
    Generate random unitaries using the Haar measure.

    Session-scoped fixture parametrized over seeds [58, 3854]
    and num_qubits [1, 2, 3], providing 12 different random unitaries.

    :param seed: Random seed for JAX PRNG
    :param num_qubits: Number of qubits for the channel
    :return: Unitary object representing a random unitary operation
    """
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    key = jax.random.PRNGKey(seed)
    return random_unitary(dims=dims, key=key, size=ensemble_size)


@pytest.fixture(scope="session")
def random_superop_channels(random_choi_channels):
    """
    Convert random Choi channels to SuperOp representation using QuTiP.

    Handles scalar and batched Choi matrices by converting each element.

    :param random_choi_channels: Choi matrices (scalar or batched)
    :return: SuperOp object with same ensemble_size as input
    """
    choi = random_choi_channels

    if choi.ensemble_size == ():
        # Scalar case
        qt_choi = choi._to_qobj()
        qt_super = qt.to_super(qt_choi)
        superop_data = jnp.array(qt_super.full())
    else:
        qobjs = choi._to_qobj()  # choi.ensemble_size array of qobjs
        superop_list = [qt.to_super(qobj).full() for qobj in qobjs.flatten()]

        # Reshape back to original ensemble shape
        superop_data = jnp.stack(superop_list).reshape(choi.ensemble_size + choi.d2)

    return SuperOp.from_matrix(superop_data, choi.dims, len(choi.ensemble_size))


@pytest.fixture(scope="session")
def random_kraus_channels(random_choi_channels):
    """
    Convert random Choi channels to Kraus representation using QuTiP.

    Handles scalar and batched Choi matrices by converting each element.

    :param random_choi_channels: Choi matrices (scalar or batched)
    :return: Kraus object with same ensemble_size as input
    """
    choi = random_choi_channels

    def _to_kraus(qobj):
        kraus_list = qt.to_kraus(qobj)
        d = choi.d[0]
        d2 = choi.d2[0]
        kraus_data = jnp.zeros((d2, d, d), dtype=jnp.complex128)
        for i, k in enumerate(reversed(kraus_list)):
            kraus_data = kraus_data.at[i].set(jnp.array(k.full()))
        return kraus_data

    if choi.ensemble_size == ():
        # Scalar case
        qt_choi = choi._to_qobj()
        kraus_data = _to_kraus(qt_choi)
    else:
        qobjs = choi._to_qobj()  # choi.ensemble_size array of qobjs
        kraus_data = jnp.asarray([_to_kraus(qobj) for qobj in qobjs.flatten()])
        kraus_data = kraus_data.reshape(choi.ensemble_size + kraus_data.shape[-3:])

    return KrausMap.from_matrix(kraus_data, choi.dims, len(choi.ensemble_size))


@pytest.fixture(scope="session")
def random_pauli_liouville_channels(random_choi_channels):
    """
    Convert random Choi channels to Pauli-Liouville representation using operator_tools.

    Handles scalar and batched Choi matrices by converting each element.

    :param random_choi_channels: Choi matrices (scalar or batched)
    :return: PauliLiouville object with same ensemble_size as input
    """
    choi = random_choi_channels

    if choi.ensemble_size == ():
        # Scalar case
        pl_data = choi2pauli_liouville(choi.matrix)  # type: ignore
    else:
        # Batched case - convert each element
        flat_shape = (-1,) + choi.matrix.shape[-2:]
        flat_choi = choi.matrix.reshape(flat_shape)

        pl_list = []
        for i in range(flat_choi.shape[0]):
            pl_data_i = choi2pauli_liouville(flat_choi[i])  # type: ignore
            pl_list.append(jnp.array(pl_data_i))

        # Reshape back to original ensemble shape
        pl_data = jnp.stack(pl_list).reshape(choi.ensemble_size + flat_choi.shape[-2:])

    return PauliLiouville.from_matrix(jnp.array(pl_data), choi.dims, len(choi.ensemble_size))


@pytest.fixture(scope="session")
def random_unitaries_chois(random_unitaries):
    """
    Convert random Unitaries to Choi representation using QuTiP.

    Handles scalar and batched Unitaries by converting each element.

    :param random_unitaries: Unitary objects (scalar or batched)
    :return: Choi object with same ensemble_size as input
    """
    unitary = random_unitaries

    if unitary.ensemble_size == ():
        # Scalar case
        qobj = unitary._to_qobj()
        qt_choi = qt.to_choi(qobj)
        choi_data = jnp.array(qt_choi.full())
    else:
        qobjs = unitary._to_qobj()  # unitary.ensemble_size array of qobjs
        choi_list = [qt.to_choi(qobj).full() for qobj in qobjs.flatten()]

        # Reshape back to original ensemble shape
        choi_data = jnp.stack(choi_list).reshape(unitary.ensemble_size + unitary.d2)

    return Choi.from_matrix(choi_data, unitary.dims, len(unitary.ensemble_size))


@pytest.fixture(scope="session")
def random_unitaries_superops(random_unitaries):
    """
    Convert random Unitaries to SuperOp representation using QuTiP.

    Handles scalar and batched Unitaries by converting each element.

    :param random_unitaries: Unitary objects (scalar or batched)
    :return: SuperOp object with same ensemble_size as input
    """
    unitary = random_unitaries

    if unitary.ensemble_size == ():
        # Scalar case
        qobj = unitary._to_qobj()
        qt_super = qt.to_super(qobj)
        superop_data = jnp.array(qt_super.full())
    else:
        qobjs = unitary._to_qobj()  # unitary.ensemble_size array of qobjs
        superop_list = [qt.to_super(qobj).full() for qobj in qobjs.flatten()]

        # Reshape back to original ensemble shape
        superop_data = jnp.stack(superop_list).reshape(unitary.ensemble_size + unitary.d2)

    return SuperOp.from_matrix(superop_data, unitary.dims, len(unitary.ensemble_size))


@pytest.fixture(scope="session")
def random_unitaries_pauli_liouvilles(random_unitaries):
    """
    Convert random Unitaries to Pauli-Liouville representation using operator_tools.

    Handles scalar and batched Unitaries by converting each element.

    :param random_unitaries: Unitary objects (scalar or batched)
    :return: PauliLiouville object with same ensemble_size as input
    """
    unitary = random_unitaries

    if unitary.ensemble_size == ():
        # Scalar case
        pauli_liouville_data = jnp.array(kraus2pauli_liouville([unitary.matrix]))
    else:
        unitaries = unitary.matrix.reshape(-1, unitary.d[0], unitary.d[1])
        pauli_liouville_data = [kraus2pauli_liouville([u]) for u in unitaries]

        # Reshape back to original ensemble shape
        pauli_liouville_data = jnp.stack(pauli_liouville_data).reshape(unitary.ensemble_size + unitary.d2)

    return PauliLiouville.from_matrix(pauli_liouville_data, unitary.dims, len(unitary.ensemble_size))


@pytest.fixture(scope="session")
def random_unitaries_kraus_maps(random_unitaries):
    """
    Convert random Unitaries to Kraus map representation using QuTiP.

    Handles scalar and batched Unitaries by converting each element.

    :param random_unitaries: Unitary objects (scalar or batched)
    :return: Kraus object with same ensemble_size as input
    """
    unitary = random_unitaries

    if unitary.ensemble_size == ():
        # Scalar case
        qobj = unitary._to_qobj()
        qt_super = qt.kraus_to_super([qobj])
        qt_kraus = qt.to_kraus(qt_super, tol=0.0)  # keep the full rank
        kraus_data = jnp.array([jnp.array(k.full()) for k in qt_kraus])
    else:
        qobjs = unitary._to_qobj()  # unitary.ensemble_size array of qobjs
        superop_list = [qt.kraus_to_super([qobj]) for qobj in qobjs.flatten()]
        kraus_list = jnp.asarray(
            [[k.full() for k in qt.to_kraus(qt_super, tol=0.0)] for qt_super in superop_list]
        )  # keep the full rank

        # Reshape back to original ensemble shape
        kraus_data = kraus_list.reshape(unitary.ensemble_size + unitary.d2)

    return KrausMap.from_matrix(kraus_data, unitary.dims, len(unitary.ensemble_size))
