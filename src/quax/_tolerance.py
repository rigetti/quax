"""Dtype- and scale-aware numerical tolerances.

A fixed absolute tolerance cannot serve both precisions JAX runs at.  At float32 the
arithmetic itself resolves nothing finer than about ``1e-7`` relative, so a threshold near
``1e-6`` is the honest floor; at float64 that same constant is nine orders of magnitude
coarser than the noise and discards physically meaningful structure.  The helpers here derive
the threshold from the working dtype and the scale of the data, so a caller gets the tightest
tolerance its arithmetic can actually support.

The form is the standard numerical-rank one, ``dimension * eps * scale``: eigenvalue error
from a Hermitian eigensolver grows with the matrix dimension and with the largest eigenvalue,
so a threshold proportional to both separates round-off from signal.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def eigenvalue_tolerance(eigenvalues: Array, dimension: int) -> Array:
    """The threshold below which a Hermitian eigenvalue is indistinguishable from round-off.

    :param eigenvalues: The spectrum, with the eigenvalues along the last axis.  Only its
        dtype and largest magnitude are used, so ensembles are handled by broadcasting.
    :param dimension: The matrix dimension the spectrum came from.
    :return: The tolerance, keeping the last axis so it broadcasts against *eigenvalues*.
    """
    eps = jnp.finfo(eigenvalues.dtype).eps
    return dimension * eps * jnp.max(jnp.abs(eigenvalues), axis=-1, keepdims=True)


def norm_tolerance(norms: Array, dimension: int) -> Array:
    """The threshold below which a Kraus operator's Frobenius norm is round-off.

    A Kraus operator built from a Choi eigenvalue has norm ``sqrt(eigenvalue)``, so the
    eigenvalue tolerance maps to its own square root on this scale.

    :param norms: The Frobenius norms, along the last axis.
    :param dimension: The Choi dimension the operators came from.
    :return: The tolerance, keeping the last axis so it broadcasts against *norms*.
    """
    eps = jnp.finfo(norms.dtype).eps
    return jnp.sqrt(dimension * eps) * jnp.max(jnp.abs(norms), axis=-1, keepdims=True)
