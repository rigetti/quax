Promotion
=========

*Embedding channels into a larger Hilbert space*

Quantum hardware rarely behaves as the idealized two-level system we use to
reason about qubits. Superconducting transmons, for example, are weakly
anharmonic oscillators whose third level :math:`|2\rangle` is only a few percent
detuned from the computational :math:`\{|0\rangle, |1\rangle\}` transition.
Modelling *leakage* out of the computational subspace therefore requires
embedding a qubit channel into a qutrit (or larger) Hilbert space. Quax calls
this operation **promotion**.

This page explains what a faithful promotion must satisfy, why there is a *family*
of valid choices rather than a single one, the three schemes Quax implements, and
why the **weighted** scheme is the default for channels. A worked qutrit example
ties the theory back to the simulator.


The promotion problem
---------------------

Let :math:`\mathcal{E}` be a quantum channel acting on a :math:`d`-dimensional
system, written in Kraus form

.. math::
   :label: eq-promote-channel

   \mathcal{E}(\rho) = \sum_i K_i\, \rho\, K_i^\dagger,
   \qquad \sum_i K_i^\dagger K_i = I_d .

We want to embed :math:`\mathcal{E}` into a larger :math:`D`-dimensional space
(:math:`D > d`) so that the embedded channel :math:`\tilde{\mathcal{E}}`:

#. **reproduces** :math:`\mathcal{E}` on the original (computational) subspace, and
#. **acts as the identity** on the complement (the leaked levels), since a qubit
   operation should not, by itself, disturb a state that has already leaked.

Write :math:`P` for the projector onto the original subspace and
:math:`P_\perp = I_D - P` for the complement. Embedding each Kraus operator by
zero-padding, :math:`\hat{K}_i = E K_i E^\dagger` with :math:`E` the
:math:`D \times d` isometry, gives

.. math::
   :label: eq-padded-tp

   \sum_i \hat{K}_i^\dagger \hat{K}_i = P .

The padded operators alone are therefore **not** trace preserving: they annihilate
the complement subspace. We must add the missing :math:`P_\perp` back.


A family of valid extensions
----------------------------

The key observation is that the complement projector can be distributed across
the Kraus operators *in any way we like*. Consider

.. math::
   :label: eq-weighted-kraus

   \tilde{K}_i = \hat{K}_i + \alpha_i P_\perp,
   \qquad \sum_i |\alpha_i|^2 = 1 .

Because the padded operator :math:`\hat{K}_i` has support only on the original
subspace and :math:`P_\perp` only on the complement, they have **disjoint
support**: :math:`\hat{K}_i^\dagger P_\perp = 0` and
:math:`P_\perp \hat{K}_i = 0`. The cross terms in
:math:`\tilde{K}_i^\dagger \tilde{K}_i` therefore vanish, and

.. math::
   :label: eq-extension-tp

   \sum_i \tilde{K}_i^\dagger \tilde{K}_i
   = \sum_i \hat{K}_i^\dagger \hat{K}_i
   + \Big(\sum_i |\alpha_i|^2\Big) P_\perp
   = P + P_\perp = I_D .

So **every** choice of weights :math:`\{\alpha_i\}` with
:math:`\sum_i |\alpha_i|^2 = 1` yields a completely positive, trace-preserving
(CPTP) extension that satisfies both requirements above. The freedom in
:math:`\{\alpha_i\}` is exactly the freedom in *how the leaked subspace is
correlated with the channel's Kraus (trajectory) decomposition*. Different choices
produce nearly identical average density matrices but markedly different behaviour
under a Monte-Carlo (trajectory) unravelling.


Three schemes
-------------

Quax exposes three points in this family.

**Coherent.** Put the entire complement on the first Kraus operator,
:math:`\alpha_0 = 1` and :math:`\alpha_{i>0} = 0`:

.. math::
   :label: eq-coherent

   \tilde{K}_0 = \hat{K}_0 + P_\perp, \qquad \tilde{K}_{i>0} = \hat{K}_i .

This fully preserves coherence between the computational and complement subspaces
and is *exact* for a unitary channel (which has a single Kraus operator). Its
drawback is that in a trajectory simulation the leaked state survives **only**
along the :math:`K_0` branch; every other branch (e.g. an :math:`X`, :math:`Y` or
:math:`Z` error) annihilates it. The fate of the leaked population thus becomes
spuriously correlated with which gate-error trajectory was sampled.

**Incoherent.** Add :math:`P_\perp` as a *separate* Kraus operator:

.. math::
   :label: eq-incoherent

   \tilde{\mathcal{E}}(\rho) = \sum_i \hat{K}_i\, \rho\, \hat{K}_i^\dagger
   + P_\perp\, \rho\, P_\perp .

This destroys all coherence between the computational and complement subspaces. It
is the physically correct extension for **measurements and resets**, where the
leaked levels genuinely decohere from the computational state, and is what Quax
uses when promoting a :class:`~quax.QuantumInstrument`.

**Weighted (default).** Distribute the complement across *every* Kraus operator in
proportion to its Frobenius norm,

.. math::
   :label: eq-weighted

   \alpha_i = \frac{\lVert K_i \rVert_F}{\sqrt{\sum_j \lVert K_j \rVert_F^2}} .

For the canonical Choi-eigenvector decomposition the Frobenius norms satisfy
:math:`\lVert K_i \rVert_F^2 = \lambda_i` (the Choi eigenvalues) and, by trace
preservation, :math:`\sum_j \lambda_j = \operatorname{Tr} J = d`. Hence

.. math::
   :label: eq-weighted-prob

   |\alpha_i|^2 = \frac{\lVert K_i \rVert_F^2}{d}
   = \operatorname{Tr}\!\Big(K_i\, \tfrac{I_d}{d}\, K_i^\dagger\Big) = q_i ,

which is exactly the probability that the **maximally mixed** computational state
branches into Kraus operator :math:`i`. Because :math:`\tilde{K}_i|{\perp}\rangle =
\alpha_i |{\perp}\rangle` for any leaked basis state :math:`|{\perp}\rangle`, the
leaked state is preserved *identically in every branch*, while branch :math:`i`
fires with the channel's own intrinsic probability :math:`q_i`. The survival of a
leaked state is therefore **decoupled** from the computational gate trajectory.


Why Quax defaults to weighted
-----------------------------

For trajectory (Monte-Carlo wavefunction) simulation — the dominant use case for
leakage modelling — the weighted scheme is the physically sensible default:

* **Trajectory decoupling.** A state that has leaked to :math:`|2\rangle` should
  propagate through the gate independently of the gate's own error channel. The
  weighted extension guarantees :math:`\tilde{K}_i|2\rangle = \alpha_i|2\rangle` in
  *every* branch, so the leaked population is never correlated with whether an
  :math:`I`, :math:`X`, :math:`Y` or :math:`Z` error was sampled.

* **No regression for gates.** A unitary has a single Kraus operator, so
  :math:`\alpha_0 = 1` and the weighted extension reduces to the coherent one
  (Eq. :eq:`eq-coherent`). Embedding ideal gates is unchanged.

* **Correct averages.** Like every member of the family in
  Eq. :eq:`eq-weighted-kraus`, the weighted extension is CPTP and agrees with the
  other schemes on the computational and complement blocks of the output density
  matrix; the schemes differ only in the computational :math:`\leftrightarrow`
  complement coherence block.

Measurements and resets remain an exception: use
:func:`~quax.promote_incoherent` for those, as Quax does automatically when
promoting a :class:`~quax.QuantumInstrument`.

.. note::

   The weighting depends on the Kraus decomposition. A :class:`~quax.KrausMap`
   is promoted using *its own* operators, whereas :class:`~quax.SuperOp`,
   :class:`~quax.Choi` and :class:`~quax.PauliLiouville` are first decomposed into
   the canonical Choi-eigenvector Kraus set. The resulting channels are CPTP and
   identical on the computational subspace, but their off-diagonal coherence blocks
   (and hence their trajectory unravellings) reflect the chosen decomposition.


Worked example: depolarizing noise on a transmon
------------------------------------------------

Consider single-qubit depolarizing noise,

.. math::
   :label: eq-depol

   \mathcal{E}(\rho) = (1-p)\,\rho + \frac{p}{3}\big(X\rho X + Y\rho Y + Z\rho Z\big),

with Kraus operators :math:`K_0 = \sqrt{1-p}\,I` and
:math:`K_{1,2,3} = \sqrt{p/3}\,\{X, Y, Z\}`. The Frobenius norms are
:math:`\lVert K_0 \rVert_F^2 = 2(1-p)` and
:math:`\lVert K_{i} \rVert_F^2 = 2p/3`, summing to :math:`2`. The weights
(Eq. :eq:`eq-weighted`) are therefore

.. math::
   :label: eq-depol-alphas

   \alpha_0 = \sqrt{1-p}, \qquad \alpha_1 = \alpha_2 = \alpha_3 = \sqrt{p/3} .

Promoting to a qutrit (:math:`d = 2 \to D = 3`) with :math:`P_\perp =
|2\rangle\!\langle 2|` gives four Kraus operators

.. math::
   :label: eq-depol-promoted

   \tilde{K}_i =
   \begin{pmatrix} K_i & 0 \\ 0 & \alpha_i \end{pmatrix},

so each branch leaves :math:`|2\rangle` in place, scaled by :math:`\alpha_i`, while
acting as the bare Pauli on the computational block. In Quax:

.. code-block:: python

   import jax.numpy as jnp
   import quax as qx

   p = 0.1
   channel = qx.depolarizing_operators(p)        # qubit KrausMap (d = 2)

   promoted = qx.promote(channel, (3,))          # weighted extension (default)

   # Each promoted Kraus operator preserves |2> with weight alpha_i:
   alphas = promoted.matrix[:, 2, 2]
   print(alphas)                                  # [sqrt(1-p), sqrt(p/3), sqrt(p/3), sqrt(p/3)]
   print(jnp.sum(jnp.abs(alphas) ** 2))           # 1.0  (CPTP)

Because the weights are normalized, the embedded channel is CPTP, and because the
leaked weight appears in every Kraus operator, a leaked :math:`|2\rangle` survives
along all four trajectories.

The schemes differ only in how the computational subspace stays coherent with the
leaked level. Apply the promoted channel to the coherent superposition
:math:`(|1\rangle + |2\rangle)/\sqrt{2}` and inspect the
:math:`\rho_{12}` element:

.. code-block:: python

   psi = jnp.array([0.0, 1.0, 1.0], dtype=complex) / jnp.sqrt(2.0)
   rho = jnp.outer(psi, psi.conj())

   weighted = qx.kraus_to_superop(qx.promote(channel, (3,)))
   incoherent = qx.promote_incoherent(qx.kraus_to_superop(channel), (3,))

   rho_w = (weighted.matrix @ rho.ravel()).reshape(3, 3)
   rho_i = (incoherent.matrix @ rho.ravel()).reshape(3, 3)

   print(abs(rho_w[1, 2]))   # > 0  : weighted keeps the computational<->leaked coherence
   print(abs(rho_i[1, 2]))   # = 0  : incoherent destroys it

The weighted (default) extension retains a partial :math:`\rho_{12}` coherence,
whereas the incoherent extension — appropriate for a measurement — removes it
entirely.


API reference
-------------

* :func:`~quax.promote` — promote a quantum object; uses the **weighted**
  extension for channels and zero-pads / identity-extends states, operators and
  unitaries.
* :func:`~quax.promote_incoherent` — the incoherent extension for measurements
  and resets.
* :func:`~quax.embed` — positional embedding of an object into specified qudits of
  a larger register (built on :func:`~quax.promote`).
