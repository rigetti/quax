"""Generate figures for the quantum instruments documentation.

Run from the docs/_static directory:
    python generate_qi_figures.py
"""

import sys

sys.path.insert(0, "../../src")

import jax.numpy as jnp

import quax as qx

# ======================================================================
# Figure 1: Qutrit instrument with confusion
# ======================================================================


def generate_confused_qutrit():
    """Generate a qutrit instrument with asymmetric confusion."""
    d = 3
    confusion = jnp.array(
        [
            [0.95, 0.04, 0.02],
            [0.04, 0.90, 0.08],
            [0.01, 0.06, 0.90],
        ]
    )
    transition = jnp.eye(d)
    qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(3,))
    fig = qx.plot(qi)
    fig.write_image("confused-qi-qutrit.png", scale=3)
    print("Generated: confused-qi-qutrit.png")


# ======================================================================
# Figure 2: Qutrit instrument with transition (backaction)
# ======================================================================


def generate_transition_qutrit():
    """Generate a qutrit instrument with cyclic transition."""
    d = 3
    p_flip = 0.15
    confusion = jnp.eye(d)
    transition = jnp.zeros((d, d))
    for j in range(d):
        transition = transition.at[j, j].set(1 - p_flip)
        transition = transition.at[(j + 1) % d, j].set(p_flip)

    qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(3,))
    fig = qx.plot(qi)
    fig.write_image("transition-qi-qutrit.png", scale=3)
    print("Generated: transition-qi-qutrit.png")


# ======================================================================
# Figure 3: Binary measurement (|2> confused for |1>)
# ======================================================================


def generate_binary_leakage_confusion():
    """Generate a qutrit instrument where |2> is misclassified as |1> (80%) or |0> (20%)."""
    d = 3
    confusion = jnp.array(
        [
            [1.0, 0.0, 0.20],
            [0.0, 1.0, 0.80],
            [0.0, 0.0, 0.00],
        ]
    )
    transition = jnp.eye(d)
    qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(3,))
    fig = qx.plot(qi)
    fig.write_image("binary-qi-qutrit.png", scale=3)
    print("Generated: binary-qi-qutrit.png")


# ======================================================================
# Figure 4: Leakage-inducing instrument
# ======================================================================


def generate_leakage_instrument():
    """Generate a qutrit instrument that induces leakage to |2>."""
    d = 3
    confusion = jnp.eye(d)
    transition = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, 0.1, 1.0],
        ]
    )
    qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(3,))
    fig = qx.plot(qi)
    fig.write_image("leakage-qi-qutrit.png", scale=3)
    print("Generated: leakage-qi-qutrit.png")


# ======================================================================
# Figure 5: Composition of instruments
# ======================================================================


def generate_composition_figure():
    """Show composition of two qubit instruments."""
    fid = 0.90
    confusion = jnp.array([[fid, 1 - fid], [1 - fid, fid]])
    noisy = qx.instrument_from_confusion_and_transition(confusion, jnp.eye(2), dims=(2,))
    ideal = qx.gates.MEASURE()
    composed = ideal @ noisy
    fig = qx.plot(composed)
    fig.write_image("composition-qi.png", scale=3)
    print("Generated: composition-qi.png")


# ======================================================================
# Figure 6: 2-qubit instrument with correlated confusion
# ======================================================================


def generate_correlated_2qubit():
    """Show a 2-qubit instrument with correlated confusion."""
    confusion = jnp.array(
        [
            [0.90, 0.04, 0.06, 0.01],
            [0.04, 0.85, 0.01, 0.08],
            [0.05, 0.01, 0.88, 0.04],
            [0.01, 0.10, 0.05, 0.87],
        ]
    )
    transition = jnp.eye(4)
    qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2, 2))
    fig = qx.plot(qi)
    fig.write_image("correlated-qi-2qubit.png", scale=3.6)
    print("Generated: correlated-qi-2qubit.png")


# ======================================================================
# Figure 7: Spectator instrument
# ======================================================================


def generate_spectator_figure():
    """Show a 2-qubit instrument measuring qubit 0 with X backaction on qubit 1."""
    X = jnp.array([[0, 1], [1, 0]], dtype=complex)
    action = jnp.kron(jnp.eye(2, dtype=complex), X)

    d_total = 4
    superop_list = []
    for k in range(2):
        proj_k_full = jnp.zeros((d_total, d_total), dtype=complex)
        for idx in range(d_total):
            i0 = idx // 2
            if i0 == k:
                proj_k_full = proj_k_full.at[idx, idx].set(1.0)
        kraus = proj_k_full @ action
        superop_k = jnp.einsum("ab,cd->acbd", jnp.conj(kraus), kraus).reshape(16, 16)
        superop_list.append(superop_k)
    matrices = jnp.stack(superop_list, axis=0)
    qi = qx.QuantumInstrument.from_matrix(matrices, ((2, 2), (2, 2)), (0,))
    fig = qx.plot(qi)
    fig.write_image("spectator-qi.png", scale=3)
    print("Generated: spectator-qi.png")


if __name__ == "__main__":
    generate_confused_qutrit()
    generate_transition_qutrit()
    generate_binary_leakage_confusion()
    generate_leakage_instrument()
    generate_composition_figure()
    generate_correlated_2qubit()
    generate_spectator_figure()
    print("\nAll figures generated successfully!")
