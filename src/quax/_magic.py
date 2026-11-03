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

from functools import singledispatch
from math import e, isfinite, isqrt, prod

import jax
import jax.numpy as jnp
from jax import Array

from ._characteristic_function import characteristic_distribution, characteristic_function
from ._fast_weyl_transforms import wigner_transform
from ._pauli_vector import square_dims
from ._promotion import promote_state_vector_to_density_matrix
from ._quantum_objects import DensityMatrix, Observable, StateVector


@singledispatch
def stabilizer_norm(state) -> Array:
    r"""The stabilizer norm :math:`\mathcal{D}(\rho)` of a state.

    .. math:: \mathcal{D}(\rho) = \frac{1}{D}\sum_{x,z} |\mathrm{Tr}[W_{x,z}\,\rho]|

    The sum runs over the :math:`D^2` Weyl operators :math:`W_{x,z}` of
    :func:`characteristic_function`, which for qubits are the Pauli strings. [CATMAGIC]_ Lemma 1,
    Eq. (19) introduces it as the st-norm :math:`\|\rho\|_{st} = \sum_j |a_j|` of the Pauli
    expansion :math:`\rho = \sum_j a_j P_j`, and [ROMFTQC]_ Eq. (22) writes it as above. Only
    :math:`|\mathrm{Tr}[W\rho]|` enters, so the phase convention of the Weyl operators does not
    matter.

    Both papers define it for qubits. For qudits of any dimension it is extended here by replacing
    the Pauli strings with the :math:`D^2` Weyl operators. The properties below carry over unchanged: a
    pure stabilizer state is a uniform combination of the :math:`D` Weyl operators in its
    stabilizer group, each with :math:`|\mathrm{Tr}[W\rho]| = 1`, and Weyl operators factor across
    subsystems.

    How to read it:

    * It equals 1 on every pure stabilizer state and is at most 1 on every mixture of stabilizer
      states, because it is convex ([CATMAGIC]_ Lemma 1; [ROMFTQC]_ D1). A value above 1 therefore
      certifies that the state is not a stabilizer state ([ROMFTQC]_ D2). The converse fails beyond
      one qubit: many magic states score at most 1. For a single qubit it is exact, marking out the
      stabilizer octahedron in the Bloch ball.
    * It is a lower bound on the robustness of magic :math:`\mathcal{R}(\rho)`, the least
      :math:`\ell_1` norm of a decomposition of :math:`\rho` into stabilizer states, which governs
      the cost of classical simulation: :math:`\mathcal{D}(\rho) \le \mathcal{R}(\rho)`, and for
      :math:`n` qubits :math:`(\mathcal{D}(\rho) - 2^{-n}) / (1 - 2^{-n}) \le \mathcal{R}(\rho)`
      ([ROMFTQC]_ D3, D3*, Eq. (23)).
    * It is multiplicative,
      :math:`\mathcal{D}(\rho_1 \otimes \rho_2) = \mathcal{D}(\rho_1)\,\mathcal{D}(\rho_2)`
      ([CATMAGIC]_ Eq. (21); [ROMFTQC]_ D4, Eq. (35)).
    * The maximally mixed state gives :math:`1/D`; the single-qubit T state
      :math:`(|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}` gives :math:`(1 + \sqrt{2})/2`.

    Computed from :func:`characteristic_function` in :math:`O(D^2 \log D)` time. Dispatches on the
    type of *state*; ensembles broadcast; the function is jitted and differentiable.

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
@jax.jit
def _density_matrix_stabilizer_norm(state: DensityMatrix) -> Array:
    chi = characteristic_function(state)
    # |chi| directly: sqrt(|chi|**2) has a NaN gradient wherever chi is zero.
    return jnp.sum(jnp.abs(chi), axis=-1) / isqrt(chi.shape[-1])


@stabilizer_norm.register(StateVector)
@jax.jit
def _state_vector_stabilizer_norm(state: StateVector) -> Array:
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

    ([SRENT]_ Eq. (1)). :math:`\Xi` is :func:`characteristic_distribution`, a probability
    distribution over the :math:`D^2` Weyl operators of :func:`characteristic_function` (for
    qubits the Pauli strings), and :math:`M_\alpha` is its Rényi entropy shifted by
    :math:`-\log D`. The limit :math:`\alpha \to 1` is the Shannon entropy, and :math:`\alpha = 0`
    counts the nonzero :math:`\Xi`. The limit :math:`\alpha \to \infty` is not offered: the
    largest weight of a pure state is always the identity's, :math:`1/D`, so :math:`M_\infty = 0`
    for every pure state ([SEMONOT]_ Sec. 4).

    How to read it:

    * It is zero exactly on pure stabilizer states and positive otherwise (faithfulness),
      unchanged by Clifford unitaries, and additive under tensor products ([SRENT]_ properties
      (i)–(iii)).
    * It is a magic monotone for every integer :math:`\alpha \ge 2`: no stabilizer protocol that
      turns a pure state into a pure state, measurements and feed-forward included, increases it
      ([SEMONO]_ Theorem 1). For :math:`0 \le \alpha < 2` it is not ([SEMONOT]_ Sec. 3.1), and for
      non-integer :math:`\alpha > 2` the question is open.
    * It is at most the stabilizer nullity ([SRENT]_ Eq. (5)). For :math:`\alpha \ge 1/2` it is at
      most twice the log of the robustness of magic, and :math:`M_{1/2} = 2 \log \mathcal{D}` for
      the :func:`stabilizer_norm` :math:`\mathcal{D}` ([SRENT]_).
    * The T state :math:`|H\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}` gives
      :math:`M_\alpha(|H\rangle^{\otimes n}) = \frac{n}{1-\alpha}(\log(2^{1-\alpha} + 1) - \log 2)`,
      so :math:`M_2(|H\rangle) = \log(4/3)`.

    The cited proofs are for qubits. [QDSRE]_ Eq. (32) defines the same quantity for qudits of any
    dimension over all :math:`D^2` Weyl operators, and asserts the three properties above without
    proof.

    The logarithm has base *base*, natural by default; ``base=2`` gives bits. For
    :math:`\alpha = 0`, weights below ``D**2 * eps * max(Xi)`` count as zero, since round-off
    leaves exact zeros at about ``eps**2``. The order *alpha* selects the computation and is static
    under ``jax.jit``.

    Defined for pure states only; a :class:`DensityMatrix` raises.

    :param state: A ``StateVector``, possibly an ensemble.
    :param alpha: Rényi order, finite and ``alpha >= 0``.
    :param base: Base of the logarithm.
    :return: Real array of shape ``(*ensemble,)``.
    :raises ValueError: If ``alpha`` is negative or not finite.

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
        tolerance = xi.shape[-1] * jnp.finfo(xi.dtype).eps * jnp.max(xi, axis=-1, keepdims=True)
        entropy = jnp.log(jnp.sum(xi > tolerance, axis=-1).astype(xi.dtype))
    else:
        entropy = jnp.log(jnp.sum(jnp.where(positive, safe**alpha, 0.0), axis=-1)) / (1 - alpha)
    return (entropy - jnp.log(isqrt(xi.shape[-1]))) / jnp.log(base)


@singledispatch
def mixed_stabilizer_renyi_entropy(state, base: float = e) -> Array:
    r"""Leone et al.'s stabilizer 2-Rényi entropy of a mixed state, :math:`\tilde M_2(\rho)`.

    .. math::

        \tilde M_2(\rho) = M_2(\rho) - S_2(\rho)
        = -\log \frac{\sum_{x,z} |\mathrm{Tr}[W_{x,z}\,\rho]|^4}{\sum_{x,z} |\mathrm{Tr}[W_{x,z}\,\rho]|^2}

    ([SRENT]_ Eq. (7); supplement Eqs. (S.29), (S.30)), with :math:`S_2(\rho) = -\log \mathrm{Tr}[\rho^2]`
    the Rényi-2 entropy and :math:`M_2(\rho) = -\log D\,\mathrm{Tr}[Q \rho^{\otimes 4}]`,
    :math:`Q = D^{-2}\sum_P P^{\otimes 4}`. The paper defines it for :math:`\alpha = 2` only. For a
    pure state :math:`S_2 = 0` and it equals ``stabilizer_renyi_entropy(state, alpha=2)``.

    How to read it:

    * It is zero exactly on the states :math:`D^{-1}(I + \sum_{P \in G} \pm P)`, for any set
      :math:`G` of Pauli strings and signs that makes this a valid state: pure stabilizer states,
      the maximally mixed state, and the normalized projectors onto stabilizer codes ([SRENT]_
      supplement, Theorem (i), Eq. (S.27)).
    * It is **not** zero on a general mixture of stabilizer states:
      :math:`(|0\rangle\langle 0| + |+\rangle\langle +|)/2` gives :math:`\log(4/3)`. So it is not
      a faithful measure or monotone for the usual resource theory, whose free states are all
      stabilizer mixtures. Use it as an indicator that is cheap to evaluate and to measure.
    * It is unchanged by Clifford unitaries and additive under tensor products ([SRENT]_
      supplement, Theorem (ii), (iii)). Numerical evidence suggests it does not increase under
      partial trace ([SRENT]_).

    The paper's proofs are for qubits; for qudits the Pauli strings are replaced by the
    :math:`D^2` Weyl operators, as in :func:`stabilizer_renyi_entropy`. The logarithm has base
    *base*, natural by default. The full citation is under :func:`stabilizer_renyi_entropy`.

    :param state: A ``DensityMatrix`` or ``StateVector``, possibly an ensemble.
    :param base: Base of the logarithm.
    :return: Real array of shape ``(*ensemble,)``.
    """
    raise TypeError(
        f"mixed_stabilizer_renyi_entropy() does not support type {type(state)!r}. "
        "Expected a DensityMatrix or StateVector."
    )


@mixed_stabilizer_renyi_entropy.register(DensityMatrix)
@jax.jit
def _density_matrix_mixed_stabilizer_renyi_entropy(state: DensityMatrix, base: float = e) -> Array:
    xi = characteristic_distribution(state)
    # sum |chi|^4 / sum |chi|^2 = D * sum xi^2 / sum xi, with xi = |chi|^2 / D.
    return -jnp.log(isqrt(xi.shape[-1]) * jnp.sum(xi**2, axis=-1) / jnp.sum(xi, axis=-1)) / jnp.log(base)


@mixed_stabilizer_renyi_entropy.register(StateVector)
@jax.jit
def _state_vector_mixed_stabilizer_renyi_entropy(state: StateVector, base: float = e) -> Array:
    xi = characteristic_distribution(state)
    # sum |chi|^4 / sum |chi|^2 = D * sum xi^2 / sum xi, with xi = |chi|^2 / D.
    return -jnp.log(isqrt(xi.shape[-1]) * jnp.sum(xi**2, axis=-1) / jnp.sum(xi, axis=-1)) / jnp.log(base)


def _require_odd(dims: tuple[int, ...]) -> None:
    if any(d % 2 == 0 for d in dims):
        raise ValueError(f"The discrete Wigner function needs odd local dimensions, got {dims}.")


@singledispatch
def wigner_function(operator) -> Array:
    r"""The discrete Wigner function of a state or observable on qudits of odd dimension.

    For a state :math:`\rho` on :math:`n` qudits,

    .. math::

        W_\rho(u) = \frac{1}{D}\,\mathrm{Tr}[A_u\,\rho],
        \qquad
        A_0 = \frac{1}{D}\sum_v T_v,
        \qquad
        A_u = T_u A_0 T_u^\dagger,

    over the :math:`D^2` phase-space points :math:`u` ([RTSC]_ Sec. II B). :math:`T_u` are the
    Weyl operators :math:`\omega^{-2^{-1}pq} Z^p X^q`, and :math:`A_u` the phase-point operators;
    :math:`A_0` is the parity operator, :math:`A_0|q\rangle = |-q\rangle`. This is the
    symplectic Fourier transform of the characteristic function ([HUDSON]_ Definition 5,
    Eq. (15), Theorem 6.5). For an observable or measurement operator :math:`E` the normalization
    is :math:`W_E(u) = \mathrm{Tr}[A_u E]`, without the :math:`1/D` ([RTSC]_ Sec. II B), so that
    :math:`\mathrm{Tr}[E\rho] = \sum_u W_E(u)\,W_\rho(u)`.

    The papers label a point :math:`u = (p, q)` by momentum (the power of :math:`Z`) and position
    (the power of :math:`X`, the computational-basis index). The result is indexed position
    first: ``W[q_0, ..., q_{n-1}, p_0, ..., p_{n-1}]``, the order of
    :func:`characteristic_function`'s :math:`(x, z)`.

    How to read it:

    * It is a real quasi-probability distribution. For a state it sums to 1, and summing over the
      momenta gives the computational-basis probabilities,
      :math:`\sum_p W_\rho(q, p) = \langle q|\rho|q\rangle` ([HUDSON]_ Theorem 6; [RTSC]_).
    * A pure state has a non-negative Wigner function if and only if it is a stabilizer state
      (the discrete Hudson theorem, [HUDSON]_ Theorem 2), and every mixture of stabilizer states
      is non-negative. Negative entries are therefore a witness of magic, measured by
      :func:`sum_negativity` and :func:`mana`.
    * Clifford unitaries act on it as affine symplectic permutations of phase space
      ([HUDSON]_ Theorem 7).
    * The qutrit state :math:`|0\rangle` is :math:`1/3` on the line :math:`q = 0` and zero
      elsewhere. The qutrit "strange state" :math:`(|1\rangle - |2\rangle)/\sqrt{2}` is
      :math:`-1/3` at the origin.

    It is defined for odd dimensions, where :math:`2^{-1} \bmod d` exists; composite odd
    dimensions and mixtures such as ``(3, 5)`` are allowed. Computed from Gross's formula
    :math:`W_\rho(q, p) = D^{-1}\sum_\xi \omega^{-\xi \cdot p}\,\rho[q + 2^{-1}\xi, q - 2^{-1}\xi]`
    ([HUDSON]_ Theorem 6.2) in :math:`O(D^2 \log D)` time. Dispatches on the type of *operator*;
    ensembles broadcast; the function is jitted and differentiable. The full citation for
    [HUDSON]_ is under :func:`characteristic_function`.

    :param operator: A ``DensityMatrix``, ``StateVector`` or ``Observable``, possibly an ensemble.
    :return: Real array of shape ``(*ensemble, *dims, *dims)``, indexed ``[q, p]``.
    :raises ValueError: If any local dimension is even.

    .. [RTSC] The resource theory of stabilizer quantum computation.
             Veitch, Mousavian, Gottesman, Emerson.
             New J. Phys. 16, 013009 (2014).
             https://doi.org/10.1088/1367-2630/16/1/013009
             https://arxiv.org/abs/1307.7171
    """
    raise TypeError(
        f"wigner_function() does not support type {type(operator)!r}. "
        "Expected a DensityMatrix, StateVector or Observable."
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
