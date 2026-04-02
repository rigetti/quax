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

import numpy as np
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
import qutip as qt  # noqa: E402

from quax import (  # noqa: E402
    Choi,
    KrausMap,
    PauliLiouville,
    SuperOp,
    choi_to_superop,
    random_choi,
    random_unitary,
)

from .reference_pauli_liouville import kraus2pauli_liouville  # noqa: E402


@pytest.fixture(scope="session", params=[58, 3854])
def seed(request):
    """Parametrized seed fixture for reproducible randomness."""
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        (2,),  # 1 qubit
        (2, 2),  # 2 qubits
        (2, 2, 2),  # 3 qubits
        (3,),  # 1 qutrit
        (3, 3),  # 2 qutrits
        (5,),  # 1 qu-5-it
        (2, 3),  # qubit-qutrit (mixed dims)
    ],
)
def dims(request):
    """Parametrized per-subsystem dimensions fixture."""
    return request.param


@pytest.fixture(scope="session", params=[(), (3,), (3, 4)])
def ensemble_size(request):
    """Parametrized ensemble size fixture."""
    return request.param


def _dims_str(dims: tuple) -> str:
    """Format a dims tuple as a filename-safe string, e.g. (2, 3) -> '2_3'."""
    return "_".join(str(d) for d in dims)


@pytest.fixture(scope="session")
def random_choi_channels(seed, dims, ensemble_size):
    """
    Generate random quantum channel as Choi matrix using BCSZ distribution.

    :param seed: Random seed for JAX PRNG
    :param dims: Per-subsystem dimensions, e.g. (2, 3) for qubit-qutrit
    :param ensemble_size: Size of the ensemble
    :return: Choi object representing a random CPTP channel
    """
    from functools import reduce
    from operator import mul

    d_total = reduce(mul, dims)
    channel_dims = (dims, dims)
    key = jax.random.PRNGKey(seed)
    return random_choi(dims=channel_dims, rank=d_total, key=key, size=ensemble_size)


@pytest.fixture(scope="session")
def save_random_choi_channels(random_choi_channels, seed, dims, ensemble_size):
    """Save random Choi channels as column-stacked superoperator matrices in npz format."""
    save_dir = Path(__file__).parent / "fixtures" / "superops"
    save_dir.mkdir(parents=True, exist_ok=True)
    ensemble_str = "_".join(str(s) for s in ensemble_size) if ensemble_size else "scalar"
    filepath = save_dir / f"superop_seed{seed}_dims{_dims_str(dims)}_ens{ensemble_str}.npz"
    # Convert Choi to superoperator matrix form
    superop = choi_to_superop(random_choi_channels)
    np.savez(
        filepath,
        data=np.asarray(superop.matrix),
        seed=np.array(seed),
        dims=np.array(dims),
        ensemble_size=np.array(ensemble_size),
    )
    return filepath


@pytest.fixture(scope="session")
def random_unitaries(seed, dims, ensemble_size):
    """
    Generate random unitaries using the Haar measure.

    :param seed: Random seed for JAX PRNG
    :param dims: Per-subsystem dimensions, e.g. (2, 3) for qubit-qutrit
    :param ensemble_size: Size of the ensemble
    :return: Unitary object representing a random unitary operation
    """
    channel_dims = (dims, dims)
    key = jax.random.PRNGKey(seed)
    return random_unitary(dims=channel_dims, key=key, size=ensemble_size)


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

    return SuperOp.from_matrix(superop_data, choi.dims)


@pytest.fixture(scope="session")
def random_kraus_channels(random_choi_channels):
    """
    Convert random Choi channels to Kraus representation using QuTiP.

    Handles scalar and batched Choi matrices by converting each element.

    :param random_choi_channels: Choi matrices (scalar or batched)
    :return: KrausMap object with same ensemble_size as input
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

    return KrausMap.from_matrix(kraus_data, choi.dims)


@pytest.fixture(scope="session")
def random_pauli_liouville_channels(random_choi_channels, seed, dims, ensemble_size):
    """
    Load pre-computed Hermitian Weyl-Liouville channels from disk.

    :param random_choi_channels: Choi matrices (used for dims)
    :param seed: Random seed
    :param dims: Per-subsystem dimensions
    :param ensemble_size: Size of the ensemble
    :return: PauliLiouville object with same ensemble_size as input
    """
    ensemble_str = "_".join(str(s) for s in ensemble_size) if ensemble_size else "scalar"
    filepath = (
        Path(__file__).parent
        / "fixtures"
        / "hermitian_weyl_liouvilles"
        / f"pauli-liouville_seed{seed}_dims{_dims_str(dims)}_ens{ensemble_str}.npz"
    )
    assert filepath.exists(), f"Reference PL file not found: {filepath.name}"
    loaded = np.load(filepath)
    pl_data = jnp.array(loaded["data"]).real
    return PauliLiouville.from_matrix(pl_data, random_choi_channels.dims)


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

    return Choi.from_matrix(choi_data, unitary.dims)


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

    return SuperOp.from_matrix(superop_data, unitary.dims)


@pytest.fixture(scope="session")
def random_unitaries_pauli_liouvilles(random_unitaries):
    """
    Convert random Unitaries to Pauli-Liouville representation using operator_tools.

    Handles scalar and batched Unitaries by converting each element.

    :param random_unitaries: Unitary objects (scalar or batched)
    :return: PauliLiouville object with same ensemble_size as input
    """
    unitary = random_unitaries

    # Reference implementation only supports qubits
    if any(d > 2 for d in unitary.dims[0]):
        pytest.skip("Reference PL implementation doesn't support qudit_dim > 2")

    if unitary.ensemble_size == ():
        # Scalar case
        pauli_liouville_data = jnp.array(kraus2pauli_liouville([unitary.matrix])).real
    else:
        unitaries = unitary.matrix.reshape(-1, unitary.d[0], unitary.d[1])
        pauli_liouville_data = [kraus2pauli_liouville([u]) for u in unitaries]

        # Reshape back to original ensemble shape
        pauli_liouville_data = jnp.stack(pauli_liouville_data).reshape(unitary.ensemble_size + unitary.d2).real

    return PauliLiouville.from_matrix(pauli_liouville_data, unitary.dims)


@pytest.fixture(scope="session")
def random_unitaries_kraus_maps(random_unitaries):
    """
    Convert random Unitaries to Kraus map representation using QuTiP.

    Handles scalar and batched Unitaries by converting each element.

    :param random_unitaries: Unitary objects (scalar or batched)
    :return: KrausMap object with same ensemble_size as input
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

    return KrausMap.from_matrix(kraus_data, unitary.dims)
