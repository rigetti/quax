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
"""Weyl characteristic functions of states and observables."""

from functools import singledispatch
from math import isqrt

import jax
import jax.numpy as jnp
from jax import Array

from ._fast_weyl_transforms import _weyl_coefficients, _xz_order
from ._operator_basis import _weyl_phases
from ._pauli_vector import _square_dims
from ._promotion import promote_state_vector_to_density_matrix
from ._quantum_objects import DensityMatrix, Observable, StateVector


def _characteristic(matrix: Array, dims: tuple[int, ...]) -> Array:
    r""":math:`\mathrm{Tr}[W_{x,z} A]` for every Weyl operator, in basis order."""
    traces = _weyl_coefficients(matrix, dims).reshape(*matrix.shape[:-2], -1)
    return jnp.take(traces, _xz_order(dims), axis=-1) * _weyl_phases(dims)


@singledispatch
def characteristic_function(operator) -> Array:
    r"""The Weyl characteristic function :math:`\chi_\rho = \mathrm{Tr}[W_{x,z}\,\rho]`.

    For :math:`n` qudits of local dimensions :math:`d_i` and :math:`D = \prod_i d_i`, the Weyl
    operators are the tensor products

    .. math::

        W_{x,z} = \bigotimes_{i} \tau_{d_i}^{x_i z_i} X_i^{x_i} Z_i^{z_i},
        \qquad
        \tau_d = (-1)^d e^{i\pi/d},
        \qquad
        0 \le x_i, z_i < d_i,

    with :math:`X_i|j\rangle = |j + 1 \bmod d_i\rangle` and :math:`Z_i|j\rangle = \omega_i^j|j\rangle`,
    :math:`\omega_i = e^{2\pi i/d_i}`. This is the phase of [SWCLIFF]_ Eq. (2.1), written with
    :math:`X` to the left of :math:`Z`. For qubits :math:`\tau = i` and the :math:`W_{x,z}` are the
    Hermitian Paulis :math:`I, X, Y, Z`, so :math:`\chi` is real and equals
    :func:`~quax.to_pauli_vector`. For odd :math:`d`, :math:`\tau = \omega^{2^{-1}}` and the
    :math:`W_{x,z}` are the Weyl operators of Gross [HUDSON]_ Eq. (3), with his :math:`(p, q)`
    being our :math:`(z, x)`. With labels reduced to :math:`0 \le x, z < d`, two identities hold
    exactly for odd :math:`d`: :math:`W_{x,z}^\dagger = W_{-x,-z}` and
    :math:`W_a W_b = \tau^{[a,b]} W_{a+b}` for the symplectic form :math:`[a, b]` ([SWCLIFF]_
    Eq. (2.4)). For even :math:`d` the phase :math:`\tau^{xz}` is not periodic modulo :math:`d`
    ([SWCLIFF]_ Eq. (2.2)), so reducing a label can flip the sign of :math:`W`. Then the
    composition law holds only up to a sign, for qubits too (:math:`YZ = iX`, where the law with
    reduced labels gives :math:`-iX`). The adjoint rule still holds exactly for qubits, but only up
    to a sign for even :math:`d > 2`.

    The :math:`D^2` values are returned as a vector in the order of
    :func:`~quax.n_qudit_basis` and :func:`~quax.weyl_basis_labels`: qudit 0 most significant, each
    qudit's :math:`(x, z)` pairs in the order of its labels. For qubits the :math:`W_{x,z}` are the
    elements of :func:`~quax.n_qudit_basis`, so :math:`\chi_k = \mathrm{Tr}[B_k \rho]` for its
    :math:`k`-th element :math:`B_k`. For :math:`d > 2` that basis holds the bare :math:`X^x Z^z`, so
    :math:`\chi_k = \tau^{xz}\,\mathrm{Tr}[B_k \rho]`: contracting :math:`\chi` with
    :func:`~quax.n_qudit_basis` does not rebuild :math:`\rho` unless the phase is applied. Since
    :math:`\mathrm{Tr}[W_a^\dagger W_b] = D\,\delta_{ab}`, the state is recovered as
    :math:`\rho = \frac{1}{D} \sum_{x,z} \chi_\rho(x,z)\, W_{x,z}^\dagger`, and
    :math:`\sum_{x,z} |\chi_\rho(x,z)|^2 = D\,\mathrm{Tr}[\rho^2]`.

    Two other normalizations go by the same name. [HUDSON]_ Eq. (14) defines
    :math:`\Xi_\rho = \mathrm{Tr}[W_{x,z}^\dagger \rho]/D`, the coefficients of :math:`\rho` in the
    Weyl basis, and [SWCLIFF]_ Eq. (2.6) defines :math:`c_\rho = \mathrm{Tr}[W_{x,z}^\dagger \rho]/\sqrt{D}`.
    For a state or observable, which is Hermitian, both are the complex conjugate of :math:`\chi`
    up to the factor::

        xi = jnp.conj(characteristic_function(rho)) / D        # Gross
        c = jnp.conj(characteristic_function(rho)) / sqrt(D)   # Gross, Nezami, Walter

    All :math:`D^2` traces are computed by gathering the shifted diagonals of :math:`\rho` and
    Fourier transforming them, in :math:`O(D^2 \log D)` time and :math:`O(D^2)` memory, without
    building the operator basis. Dispatches on the type of *operator*; ensembles broadcast; the
    function is jitted and differentiable.

    :param operator: A ``DensityMatrix``, ``StateVector`` or ``Observable``, possibly an ensemble.
    :return: Complex array of shape ``(*ensemble, D**2)``.

    .. [HUDSON] Hudson's theorem for finite-dimensional quantum systems.
             Gross.
             J. Math. Phys. 47, 122107 (2006).
             https://doi.org/10.1063/1.2393152
             https://arxiv.org/abs/quant-ph/0602001
    .. [SWCLIFF] Schur-Weyl duality for the Clifford group with applications: property testing,
             a robust Hudson theorem, and de Finetti representations.
             Gross, Nezami, Walter.
             Commun. Math. Phys. 385, 1325-1393 (2021).
             https://doi.org/10.1007/s00220-021-04118-7
             https://arxiv.org/abs/1712.08628
    """
    raise TypeError(
        f"characteristic_function() does not support type {type(operator)!r}. "
        "Expected a DensityMatrix, StateVector or Observable."
    )


@characteristic_function.register(DensityMatrix)
@jax.jit
def _density_matrix_characteristic_function(operator: DensityMatrix) -> Array:
    return _characteristic(operator.matrix, tuple(operator.dims))


@characteristic_function.register(StateVector)
@jax.jit
def _state_vector_characteristic_function(operator: StateVector) -> Array:
    return characteristic_function(promote_state_vector_to_density_matrix(operator))


@characteristic_function.register(Observable)
@jax.jit
def _observable_characteristic_function(operator: Observable) -> Array:
    return _characteristic(operator.matrix, _square_dims(operator))


@jax.jit
def characteristic_distribution(operator) -> Array:
    r"""The characteristic distribution :math:`p_\rho(x,z) = |\chi_\rho(x,z)|^2 / D`.

    [SWCLIFF]_ Eq. (3.2) defines, for a pure state, :math:`p_\psi(a) = |c_\psi(a)|^2 =
    d^{-n} |\langle\psi|W_a|\psi\rangle|^2`, which sums to one; the same formula on a general
    :math:`\rho` sums to the purity :math:`\mathrm{Tr}[\rho^2]`. For qubits it is the distribution
    :math:`\Xi_P(\psi) = \langle\psi|P|\psi\rangle^2 / d` over Pauli strings whose Rényi entropies
    are the stabilizer Rényi entropies of [SRE]_. A pure stabilizer state has :math:`p = 1/D` on the
    :math:`D` Weyl operators of its stabilizer group and zero elsewhere ([SWCLIFF]_ Eq. (3.3)), so
    any smaller maximum witnesses non-stabilizerness. The phase convention of
    :func:`characteristic_function` drops out of the modulus.

    Returned in the order of :func:`~quax.n_qudit_basis`.

    :param operator: A ``DensityMatrix``, ``StateVector`` or ``Observable``, possibly an ensemble.
    :return: Real array of shape ``(*ensemble, D**2)``.

    .. [SRE] Stabilizer Rényi entropy.
             Leone, Oliviero, Hamma.
             Phys. Rev. Lett. 128, 050402 (2022).
             https://doi.org/10.1103/PhysRevLett.128.050402
             https://arxiv.org/abs/2106.12587
    """
    chi = characteristic_function(operator)
    return jnp.abs(chi) ** 2 / isqrt(chi.shape[-1])
