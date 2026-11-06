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
"""Magic (non-stabilizerness) measures of quantum states."""

from functools import lru_cache, singledispatch
from math import e, isfinite, isqrt, prod

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from ._characteristic_function import characteristic_distribution, characteristic_function
from ._fast_weyl_transforms import wigner_transform
from ._pauli_vector import square_dims
from ._promotion import promote_state_vector_to_density_matrix
from ._quantum_objects import Choi, DensityMatrix, KrausMap, Observable, PauliLiouville, StateVector, SuperOp, Unitary
from ._superoperator_transformations import superop_in_product_basis, to_superop


@singledispatch
def stabilizer_norm(state) -> Array:
    r"""The stabilizer norm :math:`\mathcal{D}(\rho)` of a state.

    .. math:: \mathcal{D}(\rho) = \frac{1}{D}\sum_{x,z} |\mathrm{Tr}[W_{x,z}\,\rho]|

    over the :math:`D^2` Weyl operators of :func:`characteristic_function` ([CATMAGIC]_ Lemma 1;
    [ROMFTQC]_ Eq. (22)). The papers define it for qubits; quax extends it to qudits with the
    Weyl operators.

    It equals 1 on pure stabilizer states and is at most 1 on every mixture of them, so a value
    above 1 proves a state is magic. It is a witness, not a magic monotone.
    See the :doc:`magic tutorial </examples/magic>` for background and examples.

    Dispatches on the type of *state*; ensembles broadcast; jitted and differentiable.

    :param state: A ``StateVector`` or ``DensityMatrix``, possibly an ensemble.
    :return: Real array of shape ``(*ensemble,)``.

    .. [CATMAGIC] Catalysis and activation of magic states in fault-tolerant architectures.
             Campbell.
             Phys. Rev. A 83, 032317 (2011).
             https://doi.org/10.1103/PhysRevA.83.032317
             https://arxiv.org/abs/1010.0104
    .. [ROMFTQC] Application of a resource theory for magic states to fault-tolerant quantum
             computing.
             Howard, Campbell.
             Phys. Rev. Lett. 118, 090501 (2017).
             https://doi.org/10.1103/PhysRevLett.118.090501
             https://arxiv.org/abs/1609.07488
    """
    raise TypeError(
        f"stabilizer_norm() does not support type {type(state)!r}. Expected a StateVector or DensityMatrix."
    )


@stabilizer_norm.register(DensityMatrix)
@stabilizer_norm.register(StateVector)
@jax.jit
def _stabilizer_norm(state: DensityMatrix | StateVector) -> Array:
    chi = characteristic_function(state)
    # |chi| directly: sqrt(|chi|**2) has a NaN gradient wherever chi is zero.
    return jnp.sum(jnp.abs(chi), axis=-1) / isqrt(chi.shape[-1])


@singledispatch
def stabilizer_renyi_entropy(state, alpha: float = 2.0, base: float = e) -> Array:
    r"""The stabilizer Rényi entropy :math:`M_\alpha(|\psi\rangle)` of a pure state.

    .. math::

        M_\alpha(|\psi\rangle) = \frac{1}{1 - \alpha} \log \sum_{x,z} \Xi_{x,z}^\alpha - \log D,
        \qquad
        \Xi_{x,z} = \frac{|\langle\psi| W_{x,z} |\psi\rangle|^2}{D}

    ([SRENT]_ Eq. (1)), where :math:`\Xi` is :func:`characteristic_distribution`. The limit
    :math:`\alpha \to 1` is the Shannon entropy. [QDSRE]_ gives the same formula for qudits.

    It is zero exactly on pure stabilizer states, unchanged by Clifford unitaries, and additive. For
    qubits it is a magic monotone for integer :math:`\alpha \ge 2` ([SEMONO]_ Theorem 1) but not for
    :math:`\alpha < 2` ([SEMONOT]_); for qudits monotonicity is unproven.
    See the :doc:`magic tutorial </examples/magic>` for background and examples.

    Defined for pure states only; for a mixed state use :func:`mixed_stabilizer_renyi_entropy`.
    *alpha* is static under ``jax.jit``. For :math:`\alpha = 0`, weights at round-off level count
    as zero.

    :param state: A ``StateVector``, possibly an ensemble.
    :param alpha: Rényi order, finite and ``alpha >= 0``.
    :param base: Base of the logarithm; ``2`` gives bits.
    :return: Real array of shape ``(*ensemble,)``.
    :raises ValueError: If ``alpha`` is negative or not finite.
    :raises TypeError: If *state* is a ``DensityMatrix``.

    .. [SRENT] Stabilizer Rényi entropy.
             Leone, Oliviero, Hamma.
             Phys. Rev. Lett. 128, 050402 (2022).
             https://doi.org/10.1103/PhysRevLett.128.050402
             https://arxiv.org/abs/2106.12587
    .. [SEMONO] Stabilizer entropies are monotones for magic-state resource theory.
             Leone, Bittel.
             Phys. Rev. A 110, L040403 (2024).
             https://doi.org/10.1103/PhysRevA.110.L040403
             https://arxiv.org/abs/2404.11652
    .. [SEMONOT] Stabilizer entropies and nonstabilizerness monotones.
             Haug, Piroli.
             Quantum 7, 1092 (2023).
             https://doi.org/10.22331/q-2023-08-28-1092
             https://arxiv.org/abs/2303.10152
    .. [QDSRE] SWAP and transpose by displacements, stabilizer Rényi entropies for continuous
             variables and qudits and other applications.
             Klich.
             (preprint, no DOI)
             https://arxiv.org/abs/2408.15161
    """
    raise TypeError(f"stabilizer_renyi_entropy() does not support type {type(state)!r}. Expected a StateVector.")


@stabilizer_renyi_entropy.register(DensityMatrix)
def _density_matrix_stabilizer_renyi_entropy(state: DensityMatrix, alpha: float = 2.0, base: float = e) -> Array:
    raise TypeError(
        "stabilizer_renyi_entropy() is defined for pure states; pass a StateVector. "
        "Leone et al.'s mixed-state version is a separate function."
    )


@stabilizer_renyi_entropy.register(StateVector)
@jax.jit(static_argnames=("alpha",))
def _state_vector_stabilizer_renyi_entropy(state: StateVector, alpha: float = 2.0, base: float = e) -> Array:
    if alpha < 0 or not isfinite(alpha):
        raise ValueError(f"The Rényi order must be finite and non-negative, got {alpha}.")
    xi = characteristic_distribution(state)
    positive = xi > 0
    # Two wheres keep log(0) and 0**alpha out of the gradient where a weight is exactly zero.
    safe = jnp.where(positive, xi, 1.0)
    if alpha == 1:
        entropy = -jnp.sum(jnp.where(positive, xi * jnp.log(safe), 0.0), axis=-1)
    elif alpha == 0:
        # The state's precision, not xi's: a complex64 state comes back as float64 with float32 error.
        eps = jnp.finfo(state.matrix.dtype).eps
        tolerance = xi.shape[-1] * eps * jnp.max(xi, axis=-1, keepdims=True)
        entropy = jnp.log(jnp.sum(xi > tolerance, axis=-1).astype(xi.dtype))
    else:
        entropy = jnp.log(jnp.sum(jnp.where(positive, safe**alpha, 0.0), axis=-1)) / (1 - alpha)
    return (entropy - jnp.log(isqrt(xi.shape[-1]))) / jnp.log(base)


@singledispatch
def mixed_stabilizer_renyi_entropy(state, base: float = e) -> Array:
    r"""Leone et al.'s stabilizer 2-Rényi entropy of a mixed state, :math:`\tilde M_2(\rho)`.

    .. math::

        \tilde M_2(\rho)
        = -\log \frac{\sum_{x,z} |\mathrm{Tr}[W_{x,z}\,\rho]|^4}{\sum_{x,z} |\mathrm{Tr}[W_{x,z}\,\rho]|^2}

    ([SRENT]_ Eq. (7) and supplement Eq. (S.30)). On a pure state it equals
    ``stabilizer_renyi_entropy(state, alpha=2)``.

    It is zero on pure stabilizer states, the maximally mixed state and normalized
    stabilizer-code projectors, but not on general mixtures of stabilizer states, so it is not a
    magic monotone. Use it for nearly pure states.
    See the :doc:`magic tutorial </examples/magic>` for background and examples.

    :param state: A ``DensityMatrix`` or ``StateVector``, possibly an ensemble.
    :param base: Base of the logarithm.
    :return: Real array of shape ``(*ensemble,)``.
    """
    raise TypeError(
        f"mixed_stabilizer_renyi_entropy() does not support type {type(state)!r}. "
        "Expected a DensityMatrix or StateVector."
    )


@mixed_stabilizer_renyi_entropy.register(DensityMatrix)
@mixed_stabilizer_renyi_entropy.register(StateVector)
@jax.jit
def _mixed_stabilizer_renyi_entropy(state: DensityMatrix | StateVector, base: float = e) -> Array:
    xi = characteristic_distribution(state)
    # sum |chi|^4 / sum |chi|^2 = D * sum xi^2 / sum xi, with xi = |chi|^2 / D.
    return -jnp.log(isqrt(xi.shape[-1]) * jnp.sum(xi**2, axis=-1) / jnp.sum(xi, axis=-1)) / jnp.log(base)


def _require_odd(dims: tuple[int, ...]) -> None:
    if any(d % 2 == 0 for d in dims):
        raise ValueError(f"The discrete Wigner function needs odd local dimensions, got {dims}.")


@lru_cache(maxsize=32)
def _phase_point_factor(qudit_dim: int) -> npt.NDArray[np.complex128]:
    """Gross's phase-point operators ``A(p, q) = w A(0) w^dagger``, indexed ``[q * d + p]``."""
    d = qudit_dim
    omega, half = np.exp(2j * np.pi / d), (d + 1) // 2
    shift = np.roll(np.eye(d), 1, axis=0)
    clock = np.diag(omega ** np.arange(d))
    parity = np.eye(d)[(-np.arange(d)) % d]
    operators = np.empty((d * d, d, d), dtype=complex)
    for q, p in np.ndindex(d, d):
        weyl = omega ** (-half * p * q % d) * np.linalg.matrix_power(clock, p) @ np.linalg.matrix_power(shift, q)
        operators[q * d + p] = weyl @ parity @ weyl.conj().T
    return operators


@singledispatch
def wigner_function(operator) -> Array:
    r"""The discrete Wigner function of a state, observable or channel on qudits of odd dimension.

    For a state :math:`\rho`,

    .. math::

        W_\rho(u) = \frac{1}{D}\,\mathrm{Tr}[A_u\,\rho],
        \qquad
        A_u = T_u A_0 T_u^\dagger,

    over the :math:`D^2` phase-space points :math:`u`, where :math:`A_0|q\rangle = |-q\rangle` is
    the parity operator and :math:`T_u` the Weyl operators ([HUDSON]_ Eq. (15); [RTSC]_ Sec. II B).
    It is real and sums to 1. Stabilizer states and their mixtures are non-negative; negative
    entries signal magic, measured by :func:`sum_negativity` and :func:`mana`. For an observable
    :math:`E` it is :math:`\mathrm{Tr}[A_u E]`, without the :math:`1/D`, so that
    :math:`\mathrm{Tr}[E\rho] = \sum_u W_E(u)\,W_\rho(u)`.

    For a channel :math:`\mathcal{N}` it is the table

    .. math:: W_\mathcal{N}(v|u) = \frac{1}{D_\mathrm{out}}\,\mathrm{Tr}[A_v\,\mathcal{N}(A_u)]

    ([MAGCHAN]_ Definition 2), which carries Wigner functions through the channel,
    :math:`W_{\mathcal{N}(\rho)}(v) = \sum_u W_\mathcal{N}(v|u)\,W_\rho(u)`.
    See the :doc:`magic tutorial </examples/magic>` for background and examples.

    A point is indexed position first: ``W[q_0, ..., q_{n-1}, p_0, ..., p_{n-1}]``, where
    :math:`q` is the computational-basis index. A channel's table is indexed
    ``[q_out, p_out, q_in, p_in]``. Any odd dimensions are accepted, mixed ones included. An
    :class:`Involution` is treated as an observable; pass ``to_superop(gate)`` for its channel
    table. Dispatches on the type of *operator*; ensembles broadcast; jitted and differentiable.

    :param operator: A ``DensityMatrix``, ``StateVector`` or ``Observable``, or a channel
        (``Unitary``, ``SuperOp``, ``Choi``, ``KrausMap`` or ``PauliLiouville``), possibly an
        ensemble.
    :return: Real array of shape ``(*ensemble, *dims, *dims)``; for a channel,
        ``(*ensemble, *dims_out, *dims_out, *dims_in, *dims_in)``.
    :raises ValueError: If any local dimension is even.

    .. [MAGCHAN] Quantifying the magic of quantum channels.
             Wang, Wilde, Su.
             New J. Phys. 21, 103002 (2019).
             https://doi.org/10.1088/1367-2630/ab451d
             https://arxiv.org/abs/1903.04483
    .. [RTSC] The resource theory of stabilizer quantum computation.
             Veitch, Mousavian, Gottesman, Emerson.
             New J. Phys. 16, 013009 (2014).
             https://doi.org/10.1088/1367-2630/16/1/013009
             https://arxiv.org/abs/1307.7171
    """
    raise TypeError(
        f"wigner_function() does not support type {type(operator)!r}. "
        "Expected a DensityMatrix, StateVector, Observable or a channel."
    )


@wigner_function.register(DensityMatrix)
@jax.jit
def _density_matrix_wigner_function(operator: DensityMatrix) -> Array:
    dims = tuple(operator.dims)
    _require_odd(dims)
    return jnp.real(wigner_transform(operator.matrix, dims)) / prod(dims)


@wigner_function.register(StateVector)
@jax.jit
def _state_vector_wigner_function(operator: StateVector) -> Array:
    return wigner_function(promote_state_vector_to_density_matrix(operator))


@wigner_function.register(Observable)
@jax.jit
def _observable_wigner_function(operator: Observable) -> Array:
    dims = square_dims(operator)
    _require_odd(dims)
    return jnp.real(wigner_transform(operator.matrix, dims))


@wigner_function.register(SuperOp)
@wigner_function.register(Choi)
@wigner_function.register(KrausMap)
@wigner_function.register(PauliLiouville)
@wigner_function.register(Unitary)
@jax.jit
def _channel_wigner_function(operator: SuperOp | Choi | KrausMap | PauliLiouville | Unitary) -> Array:
    superop = to_superop(operator)
    dims_out, dims_in = (tuple(dims) for dims in superop.dims)
    _require_odd(dims_out + dims_in)
    lead = superop.matrix.ndim - 2
    table = superop_in_product_basis(
        superop.matrix,
        dims_out,
        dims_in,
        [_phase_point_factor(d) for d in dims_out],
        [_phase_point_factor(d) for d in dims_in],
    )
    # [(q, p) per output qudit, (q, p) per input qudit] -> [q_out, p_out, q_in, p_in]
    table = table.reshape(
        *table.shape[:-2], *(d for d in dims_out for _ in (0, 1)), *(d for d in dims_in for _ in (0, 1))
    )
    n_out, n_in = len(dims_out), len(dims_in)
    out_axes = [2 * k for k in range(n_out)] + [2 * k + 1 for k in range(n_out)]
    in_axes = [2 * n_out + 2 * k for k in range(n_in)] + [2 * n_out + 2 * k + 1 for k in range(n_in)]
    table = jnp.transpose(table, (*range(lead), *(lead + axis for axis in out_axes + in_axes)))
    return jnp.real(table) / prod(dims_out)


@singledispatch
def sum_negativity(state) -> Array:
    r"""The sum negativity :math:`\mathrm{sn}(\rho)` of a state on qudits of odd dimension.

    .. math::

        \mathrm{sn}(\rho) = \sum_{u:\,W_\rho(u) < 0} |W_\rho(u)|
        = \frac{1}{2}\Big(\sum_u |W_\rho(u)| - 1\Big)

    ([RTSC]_ Definition 10), the total negative weight of the :func:`wigner_function`.

    It is a magic monotone ([RTSC]_ Theorem 11). It is zero on every mixture of stabilizer states,
    but also on magic states whose Wigner function is non-negative. It is not additive;
    :func:`mana` is the additive version.
    See the :doc:`magic tutorial </examples/magic>` for background and examples.

    :param state: A ``DensityMatrix`` or ``StateVector``, possibly an ensemble.
    :return: Real array of shape ``(*ensemble,)``.
    :raises ValueError: If any local dimension is even.
    """
    raise TypeError(f"sum_negativity() does not support type {type(state)!r}. Expected a DensityMatrix or StateVector.")


@sum_negativity.register(DensityMatrix)
@sum_negativity.register(StateVector)
@jax.jit
def _sum_negativity(state: DensityMatrix | StateVector) -> Array:
    w = wigner_function(state)
    return jnp.sum(jnp.maximum(-w, 0), axis=tuple(range(-2 * len(state.dims), 0)))


@singledispatch
def mana(state, base: float = e) -> Array:
    r"""The mana :math:`\mathcal{M}` of a state or channel on qudits of odd dimension.

    For a state,

    .. math:: \mathcal{M}(\rho) = \log \sum_u |W_\rho(u)| = \log(2\,\mathrm{sn}(\rho) + 1)

    ([RTSC]_ Definition 12). It is an additive magic monotone ([RTSC]_ Theorem 13), zero on every
    mixture of stabilizer states, and also zero on magic states whose Wigner function is
    non-negative.

    For a channel,

    .. math:: \mathcal{M}(\mathcal{N}) = \log \max_u \sum_v |W_\mathcal{N}(v|u)|

    ([MAGCHAN]_ Definition 4), the largest mana the channel's Wigner table gives any phase-space
    point. For a quantum channel it is zero exactly when the table is non-negative, Clifford
    unitaries included.
    See the :doc:`magic tutorial </examples/magic>` for background and examples.

    :param state: A ``DensityMatrix`` or ``StateVector``, or a channel (``Unitary``, ``SuperOp``,
        ``Choi``, ``KrausMap`` or ``PauliLiouville``), possibly an ensemble.
    :param base: Base of the logarithm.
    :return: Real array of shape ``(*ensemble,)``.
    :raises ValueError: If any local dimension is even.
    """
    raise TypeError(f"mana() does not support type {type(state)!r}. Expected a state or a channel.")


@mana.register(DensityMatrix)
@mana.register(StateVector)
@jax.jit
def _mana(state: DensityMatrix | StateVector, base: float = e) -> Array:
    w = wigner_function(state)
    return jnp.log(jnp.sum(jnp.abs(w), axis=tuple(range(-2 * len(state.dims), 0)))) / jnp.log(base)


@mana.register(SuperOp)
@mana.register(Choi)
@mana.register(KrausMap)
@mana.register(PauliLiouville)
@mana.register(Unitary)
@jax.jit
def _channel_mana(state: SuperOp | Choi | KrausMap | PauliLiouville | Unitary, base: float = e) -> Array:
    superop = to_superop(state)
    dims_out, dims_in = superop.dims
    # Not wigner_function(state): an Involution would dispatch there as an Observable.
    table = _channel_wigner_function(superop)  # [*ensemble, q_out, p_out, q_in, p_in]
    lead = table.ndim - 2 * (len(dims_out) + len(dims_in))
    one_norms = jnp.sum(jnp.abs(table), axis=tuple(range(lead, lead + 2 * len(dims_out))))
    return jnp.log(jnp.max(one_norms, axis=tuple(range(lead, one_norms.ndim)))) / jnp.log(base)
