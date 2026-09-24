# This file checks our gate definitions against pyquil
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyquil.simulation import matrices

import quax as qx
from quax import is_unitary


def _num_required_positional_params(func) -> int:
    """Count required positional parameters for a callable."""
    signature = inspect.signature(func)
    return sum(
        parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in signature.parameters.values()
    )


def test_gates():
    """Check the quax gate definitions against pyquil's QUANTUM_GATES."""
    for gate_name, pyquil_matrix in matrices.QUANTUM_GATES.items():
        assert hasattr(qx.gates, gate_name), f"qx.gates is missing gate {gate_name}"
        quax_gate = getattr(qx.gates, gate_name)

        if isinstance(pyquil_matrix, np.ndarray):
            assert quax_gate.matrix.shape == pyquil_matrix.shape, f"Shape mismatch for gate {gate_name}"
            assert jnp.allclose(quax_gate.matrix, pyquil_matrix, atol=1e-6)
        else:  # parametric gate
            num_parameters = _num_required_positional_params(pyquil_matrix)
            test_params = np.linspace(0.2, 0.8, num_parameters, dtype=float) if num_parameters else np.array([])

            quax_matrix = quax_gate(*test_params).matrix
            pyquil_value = pyquil_matrix(*test_params)

            assert quax_matrix.shape == pyquil_value.shape, (
                f"Shape mismatch for gate {gate_name} with parameters {test_params}"
            )
            assert jnp.allclose(quax_matrix, pyquil_value, atol=1e-6), (
                f"Matrix mismatch for gate {gate_name} with parameters {test_params}"
            )


def test_can():
    tx = 0.73
    ty = -0.41
    tz = 0.29

    expected = jnp.array(
        [
            [
                jnp.exp(1j * tz / 2.0) * jnp.cos((tx - ty) / 2.0),
                0.0,
                0.0,
                1j * jnp.exp(1j * tz / 2.0) * jnp.sin((tx - ty) / 2.0),
            ],
            [
                0.0,
                jnp.exp(-1j * tz / 2.0) * jnp.cos((tx + ty) / 2.0),
                1j * jnp.exp(-1j * tz / 2.0) * jnp.sin((tx + ty) / 2.0),
                0.0,
            ],
            [
                0.0,
                1j * jnp.exp(-1j * tz / 2.0) * jnp.sin((tx + ty) / 2.0),
                jnp.exp(-1j * tz / 2.0) * jnp.cos((tx + ty) / 2.0),
                0.0,
            ],
            [
                1j * jnp.exp(1j * tz / 2.0) * jnp.sin((tx - ty) / 2.0),
                0.0,
                0.0,
                jnp.exp(1j * tz / 2.0) * jnp.cos((tx - ty) / 2.0),
            ],
        ],
        dtype=complex,
    )

    computed = qx.gates.CAN(tx, ty, tz).matrix
    assert jnp.allclose(computed, expected, atol=1e-6)


def test_berkeley():
    expected = jnp.array(
        [
            [jnp.cos(jnp.pi / 8.0), 0.0, 0.0, 1j * jnp.sin(jnp.pi / 8.0)],
            [0.0, jnp.cos(3.0 * jnp.pi / 8.0), 1j * jnp.sin(3.0 * jnp.pi / 8.0), 0.0],
            [0.0, 1j * jnp.sin(3.0 * jnp.pi / 8.0), jnp.cos(3.0 * jnp.pi / 8.0), 0.0],
            [1j * jnp.sin(jnp.pi / 8.0), 0.0, 0.0, jnp.cos(jnp.pi / 8.0)],
        ],
        dtype=complex,
    )

    computed = qx.gates.B.matrix
    assert jnp.allclose(computed, expected, atol=1e-6)


def test_ecr():
    expected = (1.0 / jnp.sqrt(2.0)) * jnp.array(
        [
            [0.0, 0.0, 1.0, 1j],
            [0.0, 0.0, 1j, 1.0],
            [1.0, -1j, 0.0, 0.0],
            [-1j, 1.0, 0.0, 0.0],
        ],
        dtype=complex,
    )

    computed = qx.gates.ECR.matrix
    assert jnp.allclose(computed, expected, atol=1e-6)


def test_givens():
    theta = 0.37

    expected = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, jnp.cos(theta), -jnp.sin(theta), 0.0],
            [0.0, jnp.sin(theta), jnp.cos(theta), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )

    computed = qx.gates.GIVENS(theta).matrix
    assert jnp.allclose(computed, expected, atol=1e-6)


def test_sycamore():
    expected = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1j, 0.0],
            [0.0, -1j, 0.0, 0.0],
            [0.0, 0.0, 0.0, jnp.exp(-1j * jnp.pi / 6.0)],
        ],
        dtype=complex,
    )

    computed = qx.gates.SYCAMORE.matrix
    assert jnp.allclose(computed, expected, atol=1e-6)


def test_parametric_gate_ensemble():
    """Check that a parametric gate generates an ensemble for array parameters."""
    # iterate over the parametric gates
    for gate_name, pyquil_matrix in matrices.QUANTUM_GATES.items():
        quax_gate = getattr(qx.gates, gate_name)
        if isinstance(pyquil_matrix, np.ndarray):
            continue
        else:  # parametric gate
            num_parameters = _num_required_positional_params(pyquil_matrix)

            ensemble_shape = (3, 4)
            base = np.linspace(0.1, 0.9, np.prod(ensemble_shape), dtype=float).reshape(ensemble_shape)
            params = [base + 0.1 * parameter_index for parameter_index in range(num_parameters)]

            # generate the gate
            unitaries = quax_gate(*params)

            reference_shape = pyquil_matrix(*[parameter_array[(0, 0)] for parameter_array in params]).shape
            assert unitaries.matrix.shape == ensemble_shape + reference_shape
            assert unitaries.ensemble_size == ensemble_shape

            # check that the gates match expectations using a loop
            for index in np.ndindex(*ensemble_shape):
                expected = pyquil_matrix(*[parameter_array[index] for parameter_array in params])
                computed = np.asarray(unitaries.matrix[index])
                assert jnp.allclose(computed, expected, atol=1e-6), (
                    f"Matrix mismatch for gate {gate_name} at index {index}"
                )


# =============================================================================
# Qutrit gate tests
# =============================================================================


def test_qutrit_TX_cubed_is_identity():
    """TX^3 = I for the cyclic shift gate."""
    mat = qx.gates.TX.matrix
    result = mat @ mat @ mat
    assert jnp.allclose(result, jnp.eye(3, dtype=jnp.complex128), atol=1e-10)


def test_qutrit_TZ_cubed_is_identity():
    """TZ^3 = I for the clock matrix."""
    mat = qx.gates.TZ.matrix
    result = mat @ mat @ mat
    assert jnp.allclose(result, jnp.eye(3, dtype=jnp.complex128), atol=1e-10)


def test_qutrit_TX_is_unitary():
    assert is_unitary(qx.gates.TX.matrix)


def test_qutrit_TY_is_unitary():
    assert is_unitary(qx.gates.TY.matrix)


def test_qutrit_TZ_is_unitary():
    assert is_unitary(qx.gates.TZ.matrix)


def test_qutrit_TH_is_unitary():
    """Qutrit Hadamard (QFT_3) is unitary."""
    assert is_unitary(qx.gates.TH.matrix)


def test_qutrit_TH_squared_is_not_identity():
    """QFT_3 squared is not identity (unlike qubit Hadamard)."""
    mat = qx.gates.TH.matrix
    result = mat @ mat
    assert not jnp.allclose(result, jnp.eye(3, dtype=jnp.complex128), atol=1e-6)


def test_qutrit_TH_cubed_is_identity():
    """QFT_3 cubed is identity (up to global phase)."""
    mat = qx.gates.TH.matrix
    mat3 = mat @ mat @ mat
    # QFT^n = identity up to global phase; for d=3, QFT^3 permutes but QFT^3 should give back identity
    # Actually (F_3)^4 = I. Let's check (F_3)^3 is the inverse = F_3†
    assert jnp.allclose(mat3, jnp.conj(mat.T), atol=1e-10)


def test_qutrit_rotations_are_unitary():
    """Qutrit rotations TRX, TRY, TRZ are unitary for various angles and subspaces."""
    for phi in [0.0, 0.5, jnp.pi, 2.3]:
        for axis in "XYZ":
            for subspace in ("01", "02", "12"):
                name = f"TR{axis}{subspace}"
                assert is_unitary(getattr(qx.gates, name)(phi).matrix), f"{name}({phi}) not unitary"


def test_qutrit_rotations_identity_at_zero():
    """TRX(0), TRY(0), TRZ(0) are all identity."""
    for gate in [
        qx.gates.TRX01,
        qx.gates.TRX02,
        qx.gates.TRX12,
        qx.gates.TRY01,
        qx.gates.TRY02,
        qx.gates.TRY12,
        qx.gates.TRZ01,
        qx.gates.TRZ02,
        qx.gates.TRZ12,
    ]:
        assert jnp.allclose(gate(0.0).matrix, jnp.eye(3, dtype=jnp.complex128), atol=1e-10)


def test_qutrit_TRX01_matches_qubit_RX():
    """TRX01 should match the qubit RX embedded in a 3×3 matrix."""
    phi = 1.23
    trx = qx.gates.TRX01(phi).matrix
    rx = qx.gates.RX(phi).matrix
    # Embedding: top-left 2×2 block should match RX, bottom-right is 1
    assert jnp.allclose(trx[:2, :2], rx, atol=1e-10)
    assert jnp.allclose(trx[2, 2], 1.0, atol=1e-10)


def test_qutrit_TRY01_matches_qubit_RY():
    """TRY01 should match the qubit RY embedded in a 3×3 matrix."""
    phi = 0.77
    try_mat = qx.gates.TRY01(phi).matrix
    ry = qx.gates.RY(phi).matrix
    assert jnp.allclose(try_mat[:2, :2], ry, atol=1e-10)
    assert jnp.allclose(try_mat[2, 2], 1.0, atol=1e-10)


# =============================================================================
# Gell-Mann matrix tests
# =============================================================================


def test_gellmann_are_hermitian():
    """All Gell-Mann matrices are Hermitian."""
    for i, gm in enumerate(qx.gates.GELLMANN_MATRICES):
        mat = gm.matrix
        assert jnp.allclose(mat, jnp.conj(mat.T), atol=1e-10), f"GELLMANN{i + 1} is not Hermitian"


def test_gellmann_are_traceless():
    """All Gell-Mann matrices are traceless."""
    for i, gm in enumerate(qx.gates.GELLMANN_MATRICES):
        tr = jnp.trace(gm.matrix)
        assert jnp.allclose(tr, 0.0, atol=1e-10), f"GELLMANN{i + 1} trace = {tr}, expected 0"


def test_gellmann_trace_orthogonality():
    """Gell-Mann matrices satisfy Tr(λ_i λ_j) = 2δ_{ij}."""
    for i, gi in enumerate(qx.gates.GELLMANN_MATRICES):
        for j, gj in enumerate(qx.gates.GELLMANN_MATRICES):
            tr = jnp.trace(gi.matrix @ gj.matrix)
            expected = 2.0 if i == j else 0.0
            assert jnp.allclose(tr, expected, atol=1e-10), (
                f"Tr(GELLMANN{i + 1} @ GELLMANN{j + 1}) = {tr}, expected {expected}"
            )


def test_gellmann_completeness():
    """Identity + 8 Gell-Mann matrices form a complete basis for 3×3 matrices.

    Any 3×3 matrix A can be expanded as A = (1/3)Tr(A)I + (1/2)Σ_k Tr(λ_k A) λ_k.
    """
    # Random 3×3 Hermitian matrix
    A = jnp.array([[1, 2 + 1j, 3], [2 - 1j, 4, 5 + 2j], [3, 5 - 2j, 6]], dtype=jnp.complex128)

    reconstructed = (jnp.trace(A) / 3) * jnp.eye(3, dtype=jnp.complex128)
    for gm in qx.gates.GELLMANN_MATRICES:
        reconstructed = reconstructed + 0.5 * jnp.trace(gm.matrix @ A) * gm.matrix

    assert jnp.allclose(reconstructed, A, atol=1e-10)


def test_promote_operator():
    """Test promoting a qubit gate to a qutrit."""
    # Promote Pauli X (2×2) to 3×3
    promoted = qx.promote(qx.gates.X, (3,))
    expected = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=jnp.complex128)
    assert jnp.allclose(promoted.matrix, expected, atol=1e-10)
    assert promoted.dims == ((3,), (3,))


def test_promote_operator_ensemble():
    """promote broadcasts over an ensemble of unitaries."""
    # Build an ensemble of 5 random 2×2 unitaries
    import jax

    key = jax.random.PRNGKey(0)
    ensemble = qx.random_unitary(dims=((2,), (2,)), key=key, size=(5,))
    promoted = qx.promote(ensemble, (3,))
    assert promoted.matrix.shape == (5, 3, 3)
    assert promoted.dims == ((3,), (3,))
    # Bottom-right corner must be 1 for every element
    assert jnp.allclose(promoted.matrix[:, 2, 2], 1.0, atol=1e-10)
    # Top-left 2×2 block must match original
    assert jnp.allclose(promoted.matrix[:, :2, :2], ensemble.matrix, atol=1e-10)


def test_qutrit_TRX01_ensemble():
    """TRX01 accepts an array of angles and returns an ensemble of unitaries."""
    phis = jnp.linspace(0, jnp.pi, 8)
    ensemble = qx.gates.TRX01(phis)
    assert ensemble.matrix.shape == (8, 3, 3)
    # Identity at phi=0
    assert jnp.allclose(ensemble.matrix[0], jnp.eye(3, dtype=jnp.complex128), atol=1e-8)
    # All unitary
    assert jnp.all(qx.is_unitary(ensemble))


def test_qutrit_TRY12_ensemble():
    """TRY12 accepts an array of angles and returns an ensemble of unitaries."""
    phis = jnp.array([0.0, 1.0, 2.0])
    ensemble = qx.gates.TRY12(phis)
    assert ensemble.matrix.shape == (3, 3, 3)
    assert jnp.all(qx.is_unitary(ensemble))


def test_qutrit_TRZ02_ensemble():
    """TRZ02 accepts an array of angles and returns an ensemble of unitaries."""
    phis = jnp.array([0.0, jnp.pi])
    ensemble = qx.gates.TRZ02(phis)
    assert ensemble.matrix.shape == (2, 3, 3)
    assert jnp.all(qx.is_unitary(ensemble))


def test_parametric_gates_return_unitary():
    """Parametric gates that are unitary must return Unitary instances at runtime."""
    phi, theta, lam = 0.3, 0.5, 0.7

    assert isinstance(qx.gates.RZ(phi), qx.Unitary)
    assert isinstance(qx.gates.RY(phi), qx.Unitary)
    assert isinstance(qx.gates.RX(phi), qx.Unitary)
    assert isinstance(qx.gates.PHASE(phi), qx.Unitary)
    assert isinstance(qx.gates.PHASEDRX(theta, phi), qx.Unitary)
    assert isinstance(qx.gates.U(theta, phi, lam), qx.Unitary)
    assert isinstance(qx.gates.CPHASE00(phi), qx.Unitary)
    assert isinstance(qx.gates.RZZ(phi), qx.Unitary)
    assert isinstance(qx.gates.CAN(phi, theta, lam), qx.Unitary)


def test_tensor_product_of_unitaries_is_unitary():
    """Tensor product (|) of two Unitary gates must return a Unitary."""
    phi, lam = 0.3, 0.7
    result = qx.gates.RY(phi) | qx.gates.RZ(lam)
    assert isinstance(result, qx.Unitary)


# =============================================================================
# Closed forms against the Hamiltonian generators in the gate docstrings
# =============================================================================

_I, _X, _Y, _Z = qx.gates.I, qx.gates.X, qx.gates.Y, qx.gates.Z
_II, _ZZ = _I | _I, _Z | _Z
_ZI, _IZ = _Z | _I, _I | _Z
_XX_YY = (_X | _X) + (_Y | _Y)
_YX_XY = (_Y | _X) - (_X | _Y)
_PROJ00 = (_II + _ZI + _IZ + _ZZ) * 0.25
_PROJ01 = (_II + _ZI - _IZ - _ZZ) * 0.25
_PROJ10 = (_II - _ZI + _IZ - _ZZ) * 0.25
_PROJ11 = (_II - _ZI - _IZ + _ZZ) * 0.25
_Z02 = qx.Observable.from_matrix(jnp.diag(jnp.array([1.0, 0.0, -1.0], dtype=complex)), ((3,), (3,)))
_Z12 = qx.Observable.from_matrix(jnp.diag(jnp.array([0.0, 1.0, -1.0], dtype=complex)), ((3,), (3,)))


def _phasedfsim_reference(theta, zeta, chi, gamma, phi):
    diagonal = _II * (gamma - phi / 4.0) - (_ZI + _IZ) * ((2.0 * gamma - phi) / 4.0) - _ZZ * (phi / 4.0)
    zeta_term = (_ZI - _IZ) * (zeta / 4.0)
    interaction = (_XX_YY * jnp.cos(chi) - _YX_XY * jnp.sin(chi)) * (-theta / 4.0)
    return qx.evolve(diagonal) @ qx.evolve(zeta_term) @ qx.evolve(interaction) @ qx.evolve(zeta_term)


# Each reference builds the gate from the generator(s) documented in its docstring, as exp(-iH).
_GENERATOR_REFERENCES = {
    "PHASE": lambda phi: qx.evolve((_I - _Z) * (-phi / 2.0)),
    "RX": lambda phi: qx.evolve(_X * (phi / 2.0)),
    "RY": lambda phi: qx.evolve(_Y * (phi / 2.0)),
    "RZ": lambda phi: qx.evolve(_Z * (phi / 2.0)),
    "PHASEDRX": lambda theta, phi: qx.evolve((_X * jnp.cos(phi) + _Y * jnp.sin(phi) - _I) * (theta / 2.0)),
    "U": lambda theta, phi, lam: (
        jnp.exp(0.5j * (phi + lam))
        * (qx.evolve(_Z * (phi / 2.0)) @ qx.evolve(_Y * (theta / 2.0)) @ qx.evolve(_Z * (lam / 2.0)))
    ),
    "CPHASE00": lambda phi: qx.evolve(_PROJ00 * -phi),
    "CPHASE01": lambda phi: qx.evolve(_PROJ01 * -phi),
    "CPHASE10": lambda phi: qx.evolve(_PROJ10 * -phi),
    "CPHASE": lambda phi: qx.evolve(_PROJ11 * -phi),
    "PSWAP": lambda phi: qx.evolve((_II - _ZZ) * (jnp.pi / 4.0 - phi / 2.0) - _XX_YY * (jnp.pi / 4.0)),
    "XY": lambda phi: qx.evolve(_XX_YY * (-phi / 4.0)),
    "FSIM": lambda theta, phi: qx.evolve(_XX_YY * (-theta / 4.0) - _PROJ11 * phi),
    "PHASEDFSIM": _phasedfsim_reference,
    "RXX": lambda phi: qx.evolve((_X | _X) * (phi / 2.0)),
    "RYY": lambda phi: qx.evolve((_Y | _Y) * (phi / 2.0)),
    "RZZ": lambda phi: qx.evolve(_ZZ * (phi / 2.0)),
    "CAN": lambda tx, ty, tz: qx.evolve(((_X | _X) * tx + (_Y | _Y) * ty + _ZZ * tz) * -0.5),
    "GIVENS": lambda theta: qx.evolve(_YX_XY * (theta / 2.0)),
    "TRX01": lambda phi: qx.evolve(qx.gates.GELLMANN1 * (phi / 2.0)),
    "TRY01": lambda phi: qx.evolve(qx.gates.GELLMANN2 * (phi / 2.0)),
    "TRZ01": lambda phi: qx.evolve(qx.gates.GELLMANN3 * (phi / 2.0)),
    "TRX02": lambda phi: qx.evolve(qx.gates.GELLMANN4 * (phi / 2.0)),
    "TRY02": lambda phi: qx.evolve(qx.gates.GELLMANN5 * (phi / 2.0)),
    "TRZ02": lambda phi: qx.evolve(_Z02 * (phi / 2.0)),
    "TRX12": lambda phi: qx.evolve(qx.gates.GELLMANN6 * (phi / 2.0)),
    "TRY12": lambda phi: qx.evolve(qx.gates.GELLMANN7 * (phi / 2.0)),
    "TRZ12": lambda phi: qx.evolve(_Z12 * (phi / 2.0)),
}


@pytest.mark.parametrize("gate_name", sorted(_GENERATOR_REFERENCES))
def test_closed_form_matches_generator(gate_name):
    """Each closed-form gate equals the evolution of the Hamiltonian generator documented in its docstring."""
    gate = getattr(qx.gates, gate_name)
    reference = _GENERATOR_REFERENCES[gate_name]
    num_parameters = _num_required_positional_params(gate)
    params = np.random.default_rng(1234).uniform(-2.0 * np.pi, 2.0 * np.pi, (num_parameters, 5))
    for index in range(params.shape[1]):
        args = params[:, index]
        assert jnp.allclose(gate(*args).matrix, reference(*args).matrix, atol=1e-10), f"{gate_name}{tuple(args)}"


def test_constant_gates_match_generators():
    """The fixed two-qubit gates match their documented generators."""
    assert jnp.allclose(qx.gates.ECR.matrix, (qx.evolve((_Z | _X) * (-jnp.pi / 4.0)) @ (_X | _I)).matrix, atol=1e-10)
    sycamore = qx.evolve(_XX_YY * (jnp.pi / 4.0) + _PROJ11 * (jnp.pi / 6.0))
    assert jnp.allclose(qx.gates.SYCAMORE.matrix, sycamore.matrix, atol=1e-10)
    assert jnp.allclose(qx.gates.B.matrix, qx.gates.CAN(jnp.pi / 2.0, jnp.pi / 4.0, 0.0).matrix, atol=1e-10)


def test_parametric_gate_ensembles_broadcast():
    """Parameters of different ensemble shapes broadcast against each other."""
    theta = jnp.linspace(0.1, 0.9, 3)[:, None]
    phi = jnp.linspace(-0.5, 0.5, 4)
    gates = qx.gates.FSIM(theta, phi)
    assert gates.ensemble_size == (3, 4)
    assert jnp.allclose(gates.matrix[2, 1], qx.gates.FSIM(theta[2, 0], phi[1]).matrix, atol=1e-12)


def test_parametric_gates_are_differentiable():
    """Closed forms are differentiable in their angles, matching the generator-based reference."""

    def loss(gate, *args):
        return jnp.abs(gate(*args).matrix[..., 1, 2]) ** 2 + gate(*args).matrix.real.sum()

    for gate_name in ("RX", "PHASEDFSIM", "CAN", "TRY12"):
        gate = getattr(qx.gates, gate_name)
        args = tuple(jnp.linspace(0.2, 0.8, _num_required_positional_params(gate)))
        argnums = tuple(range(len(args)))
        grads = jax.grad(lambda *a: loss(gate, *a), argnums=argnums)(*args)
        reference = jax.grad(lambda *a: loss(_GENERATOR_REFERENCES[gate_name], *a), argnums=argnums)(*args)
        assert jnp.allclose(jnp.array(grads), jnp.array(reference), atol=1e-8), gate_name
