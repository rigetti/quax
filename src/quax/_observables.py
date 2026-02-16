from ._quantum_objects import StateVector, DensityMatrix, State
from jax import Array
import jax
import jax.numpy as jnp
from functools import singledispatch


@singledispatch
def bitstring_probability(state: State, bitstring: Array) -> Array:
    """Compute the probability of measuring a specific bitstring from a state vector.

    Given a state vector |ψ⟩ and a bitstring (computational basis state index for each qudit),
    returns the probability |⟨bitstring|ψ⟩|² of measuring that bitstring.

    :param bitstring: Array of shape (num_qudits,) specifying the basis state index for each qudit.
                      For qubits, entries are 0 or 1. For qudits of dimension d, entries are 0 to d-1.
    :param state: StateVector with shape (*ensemble, d0, d1, ..., d_{n-1}) in tensor form.
    :returns: Array of shape (*ensemble,) containing the probability for each state in the ensemble.
    """
    raise NotImplementedError(f"Cannot estimate bitstring for state type {type(state)}")


@jax.jit
@bitstring_probability.register(StateVector)
def _state_bitstring_probability(state: StateVector, bitstring: Array) -> Array:
    # The state data is in tensor form: (*ensemble, d0, d1, ..., d_{n-1})
    # We need to index into the qudit dimensions using the bitstring
    # Build the index tuple: all ensemble dims get full slices, qudit dims get bitstring values
    num_ensemble_dims = state.num_ensemble_dims

    # Create index: (:, :, ..., :, b0, b1, ..., b_{n-1})
    # where there are num_ensemble_dims colons followed by bitstring indices
    idx = tuple([slice(None)] * num_ensemble_dims) + tuple(bitstring)

    # Get the amplitude for this bitstring
    amplitude = state.data[idx]

    # Return the probability |amplitude|^2
    return jnp.abs(amplitude) ** 2


@jax.jit
@bitstring_probability.register(DensityMatrix)
def _density_matrix_bitstring_probability(state: DensityMatrix, bitstring: Array) -> Array:
    # For density matrices, the probability of measuring a bitstring |b⟩ is given by
    # P(b) = ⟨b|ρ|b⟩, which corresponds to the diagonal element ρ_bb in matrix form.
    num_ensemble_dims = state.num_ensemble_dims

    # Create index for the diagonal element corresponding to the bitstring
    # (:, :, ..., :, b0, b1, ..., b_{n-1}, b0, b1, ..., b_{n-1})
    idx = tuple([slice(None)] * num_ensemble_dims) + tuple(bitstring) + tuple(bitstring)

    # Extract the diagonal element corresponding to the bitstring
    probability = state.data[idx]

    return jnp.real(probability)  # Probability should be real-valued
