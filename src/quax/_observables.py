from functools import singledispatch

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import DensityMatrix, State, StateVector


@singledispatch
def bitstring_probability(state: State, bitstring: Array) -> Array:
    """Compute the probability of measuring specific bitstring(s) from a quantum state.

    Given a state and a bitstring (computational basis state index for each qudit),
    returns the probability of measuring that bitstring.

    When *bitstring* is a 2-D array of shape ``(num_bitstrings, num_qudits)``, the
    probabilities are summed over the leading axis so a single scalar (per
    ensemble member) is returned.  This is useful for computing the probability
    of being in a subspace – e.g. pass ``jnp.array([[0], [1]])`` to get the
    total computational-subspace population of a qutrit.

    :param state: A :class:`StateVector` or :class:`DensityMatrix`.
    :param bitstring: Array of shape ``(num_qudits,)`` for a single bitstring, or
        ``(num_bitstrings, num_qudits)`` for multiple bitstrings whose probabilities
        will be summed.
    :returns: Array of shape ``(*ensemble,)`` containing the (summed) probability.
    """
    raise NotImplementedError(f"Cannot estimate bitstring for state type {type(state)}")


@jax.jit
@bitstring_probability.register(StateVector)
def _state_bitstring_probability(state: StateVector, bitstring: Array) -> Array:
    # The state data is in tensor form: (*ensemble, d0, d1, ..., d_{n-1})
    # We need to index into the qudit dimensions using the bitstring
    # Build the index tuple: all ensemble dims get full slices, qudit dims get bitstring values
    num_ensemble_dims = state.num_ensemble_dims

    if bitstring.ndim == 1:
        # Single bitstring – original path
        idx = tuple([slice(None)] * num_ensemble_dims) + tuple(bitstring)
        amplitude = state.data[idx]
        return jnp.abs(amplitude) ** 2

    # Multiple bitstrings: shape (num_bitstrings, num_qudits)
    def _prob_single(bs):
        idx = tuple([slice(None)] * num_ensemble_dims) + tuple(bs)
        amplitude = state.data[idx]
        return jnp.abs(amplitude) ** 2

    probs = jax.vmap(_prob_single)(bitstring)  # (num_bitstrings, *ensemble)
    return probs.sum(axis=0)


@jax.jit
@bitstring_probability.register(DensityMatrix)
def _density_matrix_bitstring_probability(state: DensityMatrix, bitstring: Array) -> Array:
    # For density matrices, the probability of measuring a bitstring |b⟩ is given by
    # P(b) = ⟨b|ρ|b⟩, which corresponds to the diagonal element ρ_bb in matrix form.
    num_ensemble_dims = state.num_ensemble_dims

    if bitstring.ndim == 1:
        # Single bitstring – original path
        idx = tuple([slice(None)] * num_ensemble_dims) + tuple(bitstring) + tuple(bitstring)
        probability = state.data[idx]
        return jnp.real(probability)

    # Multiple bitstrings: shape (num_bitstrings, num_qudits)
    def _prob_single(bs):
        idx = tuple([slice(None)] * num_ensemble_dims) + tuple(bs) + tuple(bs)
        probability = state.data[idx]
        return jnp.real(probability)

    probs = jax.vmap(_prob_single)(bitstring)  # (num_bitstrings, *ensemble)
    return probs.sum(axis=0)


@singledispatch
def probabilities(state: State) -> Array:
    """Compute the probabilities of all computational basis states.

    Returns a real array of shape ``(*ensemble, d)`` where ``d = prod(dims)``
    is the total Hilbert-space dimension.  Entry ``i`` is the probability of
    measuring the ``i``-th computational basis state (multi-index ravelled in
    row-major / C order over the per-qudit dimensions).

    Works for arbitrary qudit dimensions.

    :param state: A :class:`StateVector` or :class:`DensityMatrix`.
    :returns: Array of shape ``(*ensemble, d)`` with non-negative real probabilities.
    """
    raise NotImplementedError(f"Cannot compute probabilities for state type {type(state)}")


@jax.jit
@probabilities.register(StateVector)
def _state_probabilities(state: StateVector) -> Array:
    return jnp.abs(state.matrix) ** 2


@jax.jit
@probabilities.register(DensityMatrix)
def _density_matrix_probabilities(state: DensityMatrix) -> Array:
    mat = state.matrix  # (*ensemble, d, d)
    return jnp.real(jnp.diagonal(mat, axis1=-2, axis2=-1))  # (*ensemble, d)
