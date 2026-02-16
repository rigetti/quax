import jax.numpy as jnp

from ._quantum_objects import Unitary
from jax import Array


def to_euler(unitaries: Unitary) -> Array:
    """
    Convert an array of 1Q unitaries to an array of Euler angles. We use the ZXZ Euler basis, which is to say that the
    unitary will be equal to RZ(phi) @ RX(theta) @ RZ(lambda).

    :param unitaries: Unitary or unitary ensemble.
    :return: An array of Euler angles in the ZXZ basis (*ensemble_size, 3).
    """
    matrices = unitaries.matrix  # (*ensemble, 2, 2)

    # Normalize to unit determinant
    determinants = jnp.linalg.det(matrices)
    matrices = (1 / jnp.sqrt(determinants))[..., jnp.newaxis, jnp.newaxis] * matrices

    # A 2 X 2 unitary matrix has the following general form of
    # `array([[e^{-i * (lambda + phi) / 2} cos(theta / 2),
    #       -i e^{-i * (lambda - phi) / 2} sin(theta / 2)], [...]])`
    # up to an overall phase factor `e^{i * rho}` which should be ignored.
    theta = 2 * jnp.arctan2(jnp.abs(matrices[..., 1, 0]), jnp.abs(matrices[..., 0, 0]))
    angle_plus = -jnp.angle(matrices[..., 0, 0])  # lambda / 2 + phi / 2
    angle_minus = jnp.angle(matrices[..., 0, 1]) + jnp.pi / 2  # lambda / 2 - phi / 2
    lam = angle_plus + angle_minus
    phi = angle_plus - angle_minus

    euler_angles = jnp.mod(jnp.stack([phi, theta, lam], axis=-1), 2 * jnp.pi)
    return euler_angles


def to_zxz_angles(unitaries: Unitary) -> Array:
    """
    Convert an array of 1Q unitaries to an array of ZXZ angles. The angles are arranged such that::

        angles = to_zxz_angles(unitary)

        program = Program()
        program += RZ(angles[0], qubit)
        program += RX(angles[1], qubit)
        program += RZ(angles[2], qubit)

    should prepare the state transformed by unitary, i.e., `unitary @ array([1, 0])`.

    :param: unitaries: A set of unitaries to decompose (num_unitaries, 2, 2) or (2, 2).
    :return: An array of angles which can be directly used by the ZXZ decomposition (num_unitaries, 3) or (3,).
    """
    return to_euler(unitaries)[..., ::-1]


def to_zxzxz_angles(unitaries: Unitary) -> Array:
    """
    Convert an array of 1Q unitaries to an array of ZXZXZ angles. The angles are arranged such that::

        angles = to_zxzxz_angles(unitary)

        program = Program()
        program += RZ(angles[0], qubit)
        program += RX(pi/2, qubit)
        program += RZ(angles[1], qubit)
        program += RX(pi/2, qubit)
        program += RZ(angles[2], qubit)

    should prepare the state transformed by unitary, i.e., `unitary @ array([1, 0])`.

    :param: unitaries: A set of unitaries to decompose (num_unitaries, 2, 2) or (2, 2).
    :return: An array of angles which can be directly used by the ZXZXZ decomposition (num_unitaries, 3) or (3,).
    """
    return jnp.array([+1, -1, +1]) * to_euler(unitaries)[..., ::-1] + jnp.array([-jnp.pi / 2, +jnp.pi, -jnp.pi / 2])


def to_pmw3_angles(unitaries: Unitary) -> Array:
    """
    Convert an array of unitaries to PMW-3 angles.
    See arXiv:2105.02398 for details.

    That is, produce an array of angles that can be inserted into the decomposition:

    angles = to_pmw3_angles(unitary)

    phased_RX(pi/2, angles[0], qubit)
    phased_RX(pi, angles[1], qubit)
    phased_RX(pi/2, angles[2], qubit)

    These angles are related to the Euler angles, but not exactly the same.

    Note that phased_RX(theta, phase) is defined as:

    RZ(-phase) @ RX(theta) @ RZ(phase) in matrix order or

    RZ(phase)
    RX(theta)
    RZ(-phase)

    in program order.

    :param: unitaries: The unitaries to decompose (num_unitaries, 2, 2).
    :return: An array of angles which can be directly used by the PMW3 decomposition (num_unitaries, 3).
    """
    eulers = to_euler(unitaries)  # phi, theta, lam
    phis = eulers[..., 0]
    thetas = eulers[..., 1]
    lambdas = eulers[..., 2]

    alphas = -phis / 2 - lambdas / 2
    betas = phis / 2 - lambdas / 2 - jnp.pi / 2
    gammas = thetas / 2

    # reusing greek letters here
    # ω = −α − β, φ = −β + γ − π, θ = α − β
    thetas = alphas - betas
    omegas = -alphas - betas
    phis = -betas + gammas - jnp.pi
    return jnp.stack([omegas, phis, thetas], axis=-1)


def to_pmw4_angles(unitaries: Unitary) -> Array:
    """
    Convert an array of unitaries to PMW-4 angles.
    See arXiv:2105.02398 for details.

    That is, produce an array of angles that can be inserted into the decomposition:

    angles = to_pmw4_angles(unitary)

    phased_RX(pi/2, angles[0], qubit)
    phased_RX(pi/2, angles[1], qubit)
    phased_RX(pi/2, angles[1], qubit)
    phased_RX(pi/2, angles[2], qubit)

    These angles are related to the Euler angles, but not exactly the same.

    Note that phased_RX(theta, phase) is defined as:

    RZ(-phase) @ RX(theta) @ RZ(phase) in matrix order or

    RZ(phase)
    RX(theta)
    RZ(-phase)

    in program order.

    :param: unitaries: The unitaries to decompose (num_unitaries, 2, 2).
    :return: An array of angles which can be directly used by the PMW4 decomposition (num_unitaries, 3).
    """
    return to_pmw3_angles(unitaries)
