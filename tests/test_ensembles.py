import jax
import pytest
import quax as qx
import jax.numpy as jnp
from quax.ensembles import (
    ICOSAHEDRAL_ENSEMBLE,
    TETRAHEDRAL_ENSEMBLE,
    CLIFFORD_ENSEMBLE,
    SIC_PREP,
    PAULIS,
    BINARY_ICOSAHEDRAL_ENSEMBLE,
)


@pytest.mark.parametrize(
    "ensemble",
    [
        ICOSAHEDRAL_ENSEMBLE,
        TETRAHEDRAL_ENSEMBLE,
        CLIFFORD_ENSEMBLE,
        SIC_PREP,
        PAULIS,
        # BINARY_ICOSAHEDRAL_ENSEMBLE, # the binary icosahedral group is a double cover of the icosahedral
    ],
)
def test_elements_unique(ensemble):
    """Test that all elements in the ensembles are unique."""
    for i, ui in enumerate(ensemble.data):
        for j, uj in enumerate(ensemble.data):
            if i != j:
                assert not jnp.all(uj == ui)


@pytest.mark.parametrize("ensemble", [ICOSAHEDRAL_ENSEMBLE, CLIFFORD_ENSEMBLE, PAULIS, BINARY_ICOSAHEDRAL_ENSEMBLE])
def elements_complete(ensemble):
    """Test that the ensemble maps each state to each other state in the group. This is a property of groups."""
    states = [ui @ qx.zero_state_vector(1) for ui in ensemble.data]
    for j, s in enumerate(states):
        for i, u in enumerate(ensemble.data):
            psi = u @ s
            num_matches = sum([psi == sj for sj in states])
            assert num_matches >= 1, f"Element {i} from state {j}"


@pytest.mark.parametrize(
    "ensemble", [TETRAHEDRAL_ENSEMBLE, ICOSAHEDRAL_ENSEMBLE, CLIFFORD_ENSEMBLE, PAULIS, BINARY_ICOSAHEDRAL_ENSEMBLE]
)
def test_is_one_design(ensemble):
    """
    ensemble: numpy array of shape (N, 2, 2)
    """
    assert jnp.all(qx.is_one_design(ensemble, atol=1e-6))


@pytest.mark.parametrize(
    "ensemble", [TETRAHEDRAL_ENSEMBLE, ICOSAHEDRAL_ENSEMBLE, CLIFFORD_ENSEMBLE, BINARY_ICOSAHEDRAL_ENSEMBLE]
)
def test_is_two_design(ensemble):
    """
    ensemble: numpy array of shape (N, 2, 2)
    """
    assert jnp.all(qx.is_two_design(ensemble, atol=1e-6))


def test_ensembles_are_attributes_not_functions():
    assert isinstance(qx.ensembles.PAULI_ENSEMBLE, qx.Unitary)
    assert not callable(qx.ensembles.CLIFFORD_ENSEMBLE)
    assert "ICOSAHEDRAL_GROUP" in dir(qx.ensembles)
    with pytest.raises(AttributeError):
        _ = qx.ensembles.NOT_A_REAL_ENSEMBLE


def test_ensembles_use_current_precision():
    """Lazily-built ensembles are constructed at the active precision (``complex128`` under x64).

    Precision-*change* behaviour is covered in isolation by
    ``test_gates.py::test_lazy_constants_track_precision_change`` (a subprocess test, to avoid
    clearing JAX's compilation caches mid-suite).
    """
    expected = jax.dtypes.canonicalize_dtype(jnp.complex128)
    assert qx.ensembles.PAULI_ENSEMBLE.matrix.dtype == expected
    assert qx.ensembles.CLIFFORD_ENSEMBLE.matrix.dtype == expected
    assert qx.ensembles.ICOSAHEDRAL_GROUP.matrix.dtype == expected
    assert qx.ensembles.SIC_PREP.matrix.dtype == expected
