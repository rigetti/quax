# This file checks our gate definitions against pyquil
import inspect
import numpy as np
import jax.numpy as jnp
from pyquil.simulation import matrices
import quax as qx


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
