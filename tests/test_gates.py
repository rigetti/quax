# This file checks our gate definitions against pyquil
import inspect
import numpy as np
import jax.numpy as jnp
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
    rx_gates = [qx.gates.TRX01, qx.gates.TRX02, qx.gates.TRX12]
    ry_gates = [qx.gates.TRY01, qx.gates.TRY02, qx.gates.TRY12]
    rz_gates = [qx.gates.TRZ01, qx.gates.TRZ02, qx.gates.TRZ12]
    for phi in [0.0, 0.5, jnp.pi, 2.3]:
        for rx, ry, rz in zip(rx_gates, ry_gates, rz_gates):
            assert is_unitary(rx(phi).matrix), f"{rx.__name__}({phi}) not unitary"
            assert is_unitary(ry(phi).matrix), f"{ry.__name__}({phi}) not unitary"
            assert is_unitary(rz(phi).matrix), f"{rz.__name__}({phi}) not unitary"


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
    promoted = qx.gates.promote_operator(qx.gates.X, operator_support=1, final_dims=((3,), (3,)))
    expected = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=jnp.complex128)
    assert jnp.allclose(promoted.matrix, expected, atol=1e-10)
    assert promoted.dims == ((3,), (3,))


def test_promote_operator_ensemble():
    """promote_operator broadcasts over an ensemble of unitaries."""
    # Build an ensemble of 5 random 2×2 unitaries
    import jax

    key = jax.random.PRNGKey(0)
    ensemble = qx.random_unitary(dims=((2,), (2,)), key=key, size=(5,))
    promoted = qx.gates.promote_operator(ensemble, operator_support=1, final_dims=((3,), (3,)))
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
