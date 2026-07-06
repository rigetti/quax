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

"""Generator engines for :class:`~quax.Lindbladian`.

Builds the d²×d² GKSL master-equation generator from a Hamiltonian and jump operators, and
provides related helpers (e.g. converting a unitary to its Hamiltonian generator).  Kept out of
``_quantum_objects`` so the class definition stays lean; imported lazily by
:attr:`quax.Lindbladian.matrix`.
"""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from ._quantum_objects import Lindbladian, Observable, Operator, Unitary


def unitary_to_hamiltonian(unitary: Unitary) -> Observable:
    """Return the Hamiltonian generator :math:`H` (Hermitian) with :math:`e^{-iH} = U`.

    This is the inverse of the Schrödinger propagator ``evolve(Observable) -> Unitary`` at
    ``t = 1`` (:func:`~quax.evolve` uses ``exp(-i t H)``).  The principal branch is used, so
    ``H`` has eigenvalues in :math:`(-\\pi, \\pi]`.

    Differentiable with respect to ``U`` (uses ``jax.lax.linalg.eig`` with
    ``enable_eigvec_derivs=True``); gradients are well-defined where ``U`` has a non-degenerate
    spectrum — the usual caveat of eigenvector derivatives.

    :param unitary: The unitary ``U``.
    :return: The Hermitian generator ``H`` (an :class:`~quax.Observable`) satisfying ``exp(-iH) = U``.
    """
    from ._quantum_objects import Observable

    eigvals, eigvecs = jax.lax.linalg.eig(unitary.matrix, compute_left_eigenvectors=False, enable_eigvec_derivs=True)
    # exp(-i h) = λ = exp(i·arg λ)  ⇒  h = -arg(λ)  (real eigenvalues of H)
    h_eig = -jnp.angle(eigvals)
    hamiltonian = (eigvecs * h_eig[..., None, :]) @ jnp.linalg.inv(eigvecs)  # V diag(h) V⁻¹
    hamiltonian = 0.5 * (hamiltonian + jnp.conj(jnp.swapaxes(hamiltonian, -1, -2)))  # enforce Hermiticity
    return Observable.from_matrix(hamiltonian, unitary.dims)


def gate_plus_lindbladian(gate: Unitary, lind: Lindbladian) -> Lindbladian:
    """Fold a gate ``U`` into a Lindbladian as a coherent term, returning a new generator.

    The gate contributes ``-i[H_U, ·]`` with ``H_U = unitary_to_hamiltonian(U)`` (so evolving the
    result for unit time reproduces ``U`` alongside the dissipation).  If the gate and the
    Lindbladian act on different (per-subsystem) dimensions, both are promoted to the common
    (element-wise larger) dimensions first — e.g. a qubit ``CZ`` combined with qutrit leakage.

    :param gate: The gate unitary.
    :param lind: The Lindbladian carrying the jump operators (and any existing Hamiltonian).
    :return: A :class:`~quax.Lindbladian` whose Hamiltonian is ``H_U + lind.hamiltonian``.
    """
    from ._promotion import promote
    from ._quantum_objects import Lindbladian

    gate_dims, noise_dims = gate.dims[0], lind.dims[0]
    if len(gate_dims) != len(noise_dims):
        raise ValueError(
            f"Cannot combine a gate on {gate_dims} qudits with a Lindbladian on {noise_dims} qudits: "
            "the subsystem counts differ.  Tensor the noise to match (e.g. `leakage() | leakage()`)."
        )
    target = tuple(max(g, n) for g, n in zip(gate_dims, noise_dims))
    if gate_dims != target:
        gate = promote(gate, target)
    if noise_dims != target:
        lind = promote(lind, target)

    hamiltonian = unitary_to_hamiltonian(gate)
    if lind.hamiltonian is not None:
        hamiltonian = hamiltonian + lind.hamiltonian  # Observable + Observable → Observable
    return Lindbladian(hamiltonian=hamiltonian, jump_operators=lind.jump_operators)


def reconstruct_gksl(gen_matrix: Array, d: int) -> Tuple[Array, Array]:
    """Recover a canonical GKSL representation ``(H, {L_k})`` from a ``d²×d²`` generator matrix.

    Inverse of :func:`gksl_generator` (single, un-batched generator; vmap externally for ensembles).
    Uses the traceless-jump-operator gauge: the dissipator's Kossakowski matrix is the reshuffled
    generator projected onto the traceless subspace; its eigendecomposition yields the jump
    operators, and the remaining δ-structured part fixes the traceless Hamiltonian.  Negative
    Kossakowski eigenvalues (from a non-CP-generating input) are clamped, so the result is the
    nearest valid generator.

    :param gen_matrix: ``(d², d²)`` generator matrix (single, un-batched).
    :param d: Hilbert-space dimension.
    :return: ``(H, jump_ops)`` with shapes ``(d, d)`` and ``(d², d, d)``.
    """
    tensor = gen_matrix.reshape(d, d, d, d)  # [a, c, b, d']
    reshuffled = jnp.transpose(tensor, (0, 2, 1, 3)).reshape(d * d, d * d)  # R[(a,b),(c,d')]
    omega = jnp.eye(d, dtype=complex).reshape(d * d)  # vec(I)
    proj = jnp.eye(d * d, dtype=complex) - jnp.outer(omega, jnp.conj(omega)) / d
    kossakowski = proj @ reshuffled @ proj  # PSD in the traceless subspace
    kossakowski = 0.5 * (kossakowski + kossakowski.conj().T)
    evals, evecs = jnp.linalg.eigh(kossakowski)
    evals = jnp.maximum(evals.real, 0.0)  # clamp numerical/negative eigenvalues
    weighted = evecs * jnp.sqrt(evals)[None, :]  # columns are √λ_k · v_k
    jump_ops = jnp.conj(jnp.transpose(weighted).reshape(d * d, d, d))  # (n_ops=d², d, d)

    g_matrix = jnp.einsum("kca,kcb->ab", jnp.conj(jump_ops), jump_ops)  # Σ L_k† L_k
    tau = -0.5 * jnp.trace(g_matrix)
    delta = (reshuffled - kossakowski).reshape(d, d, d, d)  # purely δ-structured
    m_matrix = (jnp.einsum("aacd->cd", delta) - tau * jnp.eye(d, dtype=complex)) / d
    hamiltonian = 1j * (m_matrix + 0.5 * g_matrix)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    return hamiltonian, jump_ops


@jax.jit
def gksl_generator(hamiltonian: Observable | None, jump_operators: Operator) -> Array:
    """GKSL generator matrix ``-i[H,ρ] + Σ_k D[L_k]`` as ``(*ensemble, d², d²)``.

    Rates are pre-absorbed into ``jump_operators``. Engine for :attr:`quax.Lindbladian.matrix`.
    """
    dims = jump_operators.dims
    d = reduce(mul, dims[1], 1)
    I = jnp.eye(d, dtype=complex)
    L = jump_operators.matrix  # (*ensemble, n_ops, d, d)
    ensemble_shape = L.shape[:-3]

    # Build the GKSL generator as a rank-4 tensor with indices
    # [out_bra a, out_ket c, in_bra b, in_ket d], reshaped at the end to a (d², d²)
    # superoperator acting on vec(ρ).
    #
    # Tensor products via einsum: an einsum whose output indices are the union of the
    # (disjoint) input indices, with none summed away, is an outer/tensor product.  E.g.
    # ``einsum("ab,cd->acbd", A, B)`` computes A ⊗ B and then interleaves the axes so that
    # (a, c) form the superoperator's row multi-index and (b, d) its column multi-index —
    # exactly the layout a (d², d²) matrix on vec(ρ) needs.  ``einsum("...kab,...kcd->...acbd")``
    # is the same tensor product but additionally summed over the jump index k.

    # No-jump rate operator  G = Σ_k L_k† L_k  (Hermitian):
    #   G_{ab} = Σ_{k,c} conj(L_k)_{ca} (L_k)_{cb}
    G = jnp.einsum("...kca,...kcb->...ab", jnp.conj(L), L)
    # Jump (dissipative gain) term  Σ_k L_k ρ L_k†  — tensor product Σ_k conj(L_k) ⊗ L_k:
    #   sandwich_{acbd} = Σ_k conj(L_k)_{ab} (L_k)_{cd}
    sandwich = jnp.einsum("...kab,...kcd->...acbd", jnp.conj(L), L)
    # Two δ-structured halves of the anticommutator  −½{G, ρ} = −½(G ρ + ρ G), each an I ⊗ G tensor product:
    #   G_rho_{acbd} = δ_{ab} G_{cd}          (the G ρ half)
    G_rho = jnp.einsum("ab,...cd->...acbd", I, G)
    #   rho_G_{acbd} = conj(G)_{ab} δ_{cd}    (the ρ G half; G Hermitian ⇒ conj(G)_{ab} = G_{ba})
    rho_G = jnp.einsum("...ab,cd->...acbd", jnp.conj(G), I)
    # Dissipator  D[ρ] = Σ_k ( L_k ρ L_k† − ½{L_k† L_k, ρ} )
    gen_data = sandwich - 0.5 * G_rho - 0.5 * rho_G

    if hamiltonian is not None:
        H = hamiltonian.matrix
        # Two δ-structured halves of the commutator  −i[H, ρ] = −i(H ρ − ρ H), each an I ⊗ H tensor product:
        #   H_rho_{acbd} = δ_{ab} H_{cd}          (the H ρ half)
        H_rho = jnp.einsum("ab,...cd->...acbd", I, H)
        #   rho_H_{acbd} = conj(H)_{ab} δ_{cd}    (the ρ H half; H Hermitian ⇒ conj(H)_{ab} = H_{ba})
        rho_H = jnp.einsum("...ab,cd->...acbd", jnp.conj(H), I)
        gen_data = gen_data + (-1j * (H_rho - rho_H))

    return gen_data.reshape(ensemble_shape + (d * d, d * d))
