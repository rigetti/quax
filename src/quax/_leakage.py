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

r"""Leakage and seepage rates of a channel on qudits :cite:`WG18`.

The computational subspace of each qudit is its lowest levels (two by default), and the
computational subspace of the register, of dimension :math:`d_1`, is their tensor product,
with projector :math:`\mathbb{1}_1`; the leakage subspace is its complement,
:math:`\mathbb{1}_2 = \mathbb{1} - \mathbb{1}_1`, of dimension :math:`d_2`. The leakage rate is the
population a channel moves out of the computational subspace, averaged over its states, and the
seepage rate the population it returns, averaged over the leaked states:

.. math::

    L_1(\mathcal{E}) = \mathrm{Tr}[\mathbb{1}_2\,\mathcal{E}(\mathbb{1}_1/d_1)], \qquad
    L_2(\mathcal{E}) = \mathrm{Tr}[\mathbb{1}_1\,\mathcal{E}(\mathbb{1}_2/d_2)].

For several qudits the leakage subspace splits into the subspaces :math:`\mathcal{Y}` in which a
given set of qudits is leaked and the others are not, and the rates resolve into
:math:`L_1^{(\mathcal{Y})} = \mathrm{Tr}[\mathbb{1}_{\mathcal{Y}}\,\mathcal{E}(\mathbb{1}_1/d_1)]` and
:math:`L_2^{(\mathcal{Y})} = \mathrm{Tr}[\mathbb{1}_1\,\mathcal{E}(\mathbb{1}_{\mathcal{Y}}/d_{\mathcal{Y}})]`
(``leaked=``), whose leakage rates sum to :math:`L_1`. The population one qudit leaks, whatever
the others do, is the sum of :math:`L_1^{(\mathcal{Y})}` over the subspaces in which it is leaked.
"""

from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ._apply import apply_superop_to_density_matrix
from ._promotion import promote
from ._quantum_objects import DensityMatrix, SuperOperator, Unitary
from ._superoperator_transformations import to_superop


def _dims(channel: SuperOperator | Unitary) -> tuple[int, ...]:
    dims_out, dims_in = channel.dims
    if tuple(dims_out) != tuple(dims_in):
        raise ValueError(f"the channel must map a space to itself, not {dims_in} to {dims_out}")
    return tuple(int(d) for d in dims_out)


def _subspace_dims(dims: tuple[int, ...], subspace_dims: tuple[int, ...] | None) -> tuple[int, ...]:
    subspace = (2,) * len(dims) if subspace_dims is None else tuple(subspace_dims)
    if len(subspace) != len(dims) or any(not 0 < s <= d for s, d in zip(subspace, dims)):
        raise ValueError(f"computational subspace {subspace} does not fit in qudits with dims {dims}")
    return subspace


def _subspace_projector(
    dims: tuple[int, ...], subspace_dims: tuple[int, ...] | None = None, leaked: tuple[int, ...] | None = None
) -> np.ndarray:
    r"""The diagonal of the projector onto a computational or leakage subspace of a register of qudits.

    Built with NumPy from static arguments, so that it is a constant under ``jit``.

    :param dims: The dimension of each qudit.
    :param subspace_dims: The dimension of each qudit's computational subspace, its lowest levels;
        two per qudit by default.
    :param leaked: The qudits that are leaked. ``None`` or empty gives the computational subspace
        :math:`\mathbb{1}_1`; otherwise the subspace :math:`\mathbb{1}_{\mathcal{Y}}` in which exactly
        these qudits are outside their computational subspace and the others inside it.
    :return: The diagonal, of length ``prod(dims)``.
    """
    subspace = _subspace_dims(dims, subspace_dims)
    leaked_set = set(leaked or ())
    if not leaked_set <= set(range(len(dims))):
        raise ValueError(f"leaked qudits {sorted(leaked_set)} are not among the {len(dims)} qudits")
    factors = [
        np.where(np.arange(d) < s, i not in leaked_set, i in leaked_set).astype(float)
        for i, (d, s) in enumerate(zip(dims, subspace))
    ]
    return reduce(np.kron, factors)


def _leakage_projector(dims: tuple[int, ...], subspace_dims: tuple[int, ...] | None, leaked) -> np.ndarray:
    """The diagonal of the whole leakage subspace, or of the subspace in which exactly *leaked* are leaked."""
    if leaked is None:
        return 1.0 - _subspace_projector(dims, subspace_dims)
    if not tuple(leaked):
        raise ValueError("leaked must name at least one qudit; the empty set is the computational subspace")
    return _subspace_projector(dims, subspace_dims, tuple(leaked))


def _population(
    channel: SuperOperator | Unitary, prepared: np.ndarray, measured: np.ndarray, dims: tuple[int, ...]
) -> Array:
    """Tr[diag(measured) E(diag(prepared) / Tr)], over the channel's ensemble, for projector diagonals."""
    rho = DensityMatrix.from_matrix(jnp.diag(jnp.asarray(prepared / prepared.sum(), dtype=complex)), dims)
    output = apply_superop_to_density_matrix(to_superop(channel), rho).matrix
    return jnp.real(jnp.einsum("i,...ii->...", jnp.asarray(measured), output))


@partial(jax.jit, static_argnames=("leaked", "subspace_dims"))
def leakage_rate(
    channel: SuperOperator | Unitary,
    leaked: tuple[int, ...] | None = None,
    subspace_dims: tuple[int, ...] | None = None,
) -> Array:
    r"""The leakage rate of a channel :cite:`WG18`.

    :math:`L_1 = \mathrm{Tr}[\mathbb{1}_2\,\mathcal{E}(\mathbb{1}_1/d_1)]`: the population that leaves
    the computational subspace, averaged over its states. With *leaked*, the rate
    :math:`L_1^{(\mathcal{Y})}` into the subspace in which exactly those qudits are leaked.

    :param channel: The channel, on qudits.
    :param leaked: The qudits leaked in the target subspace; all of the leakage subspace by default.
    :param subspace_dims: The dimension of each qudit's computational subspace; two by default.
    :return: The leakage rate, a scalar or an array over the channel's ensemble.
    """
    dims = _dims(channel)
    computational = _subspace_projector(dims, subspace_dims)
    return _population(channel, computational, _leakage_projector(dims, subspace_dims, leaked), dims)


@partial(jax.jit, static_argnames=("leaked", "subspace_dims"))
def seepage_rate(
    channel: SuperOperator | Unitary,
    leaked: tuple[int, ...] | None = None,
    subspace_dims: tuple[int, ...] | None = None,
) -> Array:
    r"""The seepage rate of a channel :cite:`WG18`.

    :math:`L_2 = \mathrm{Tr}[\mathbb{1}_1\,\mathcal{E}(\mathbb{1}_2/d_2)]`: the population that returns
    to the computational subspace, averaged over the leaked states. With *leaked*, the rate
    :math:`L_2^{(\mathcal{Y})}` out of the subspace in which exactly those qudits are leaked.

    :param channel: The channel, on qudits.
    :param leaked: The qudits leaked in the source subspace; all of the leakage subspace by default.
    :param subspace_dims: The dimension of each qudit's computational subspace; two by default.
    :return: The seepage rate, a scalar or an array over the channel's ensemble.
    :raises ValueError: If the qudits have no leakage subspace, so that the rate is undefined.
    """
    dims = _dims(channel)
    leakage = _leakage_projector(dims, subspace_dims, leaked)
    if not leakage.any():
        raise ValueError(f"qudits with dims {dims} have no leakage subspace, so no seepage rate")
    return _population(channel, leakage, _subspace_projector(dims, subspace_dims), dims)


def _subspace_process_fidelity(channel: SuperOperator | Unitary, target: Unitary) -> Array:
    r"""The process fidelity of a channel to a unitary on fewer levels, on its computational subspace :cite:`WG18`.

    :math:`F = \mathrm{Tr}[(\mathbb{1}_1 \otimes \mathbb{1}_1)\,\mathcal{S}]/d_1^2`, with :math:`\mathcal{S}`
    the superoperator of the error channel :math:`\mathcal{U}^\dagger\circ\mathcal{E}` and the computational
    subspace the levels the target acts on.
    """
    dims = _dims(channel)
    subspace_dims = tuple(int(d) for d in target.dims[0])
    if len(subspace_dims) != len(dims) or any(s > d for s, d in zip(subspace_dims, dims)):
        raise ValueError(f"a target on dims {subspace_dims} does not fit in a channel on dims {dims}")
    error = to_superop(promote(target, dims).h).matrix @ to_superop(channel).matrix
    computational = _subspace_projector(dims, subspace_dims)
    weights = jnp.asarray(np.kron(computational, computational))
    return jnp.real(jnp.einsum("...ii,i->...", error, weights)) / computational.sum() ** 2
