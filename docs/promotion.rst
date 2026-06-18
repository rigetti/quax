Promotion
=========

*Embedding quantum objects into a larger Hilbert space*

Quantum hardware rarely behaves as the idealized two-level system we use to reason
about qubits. Superconducting transmons, for example, are weakly anharmonic
oscillators whose third level :math:`|2\rangle` is only a few percent detuned from
the computational :math:`\{|0\rangle, |1\rangle\}` transition. Modelling *leakage*
out of the computational subspace therefore requires embedding qubit states,
gates, and noise channels into a qutrit (or larger) Hilbert space. Quax calls this
operation **promotion** and exposes it through a single dispatched entry point,
:func:`~quax.promote`.

Throughout, fix a system of :math:`n` subsystems with computational dimensions
:math:`(d_1, \dots, d_n)` that we promote to larger dimensions
:math:`(D_1, \dots, D_n)` with :math:`D_k \ge d_k`. Write :math:`d = \prod_k d_k`
and :math:`D = \prod_k D_k` for the total dimensions. For each subsystem the
**embedding isometry** :math:`E_k \in \mathbb{C}^{D_k \times d_k}` is the
zero-padded identity :math:`(E_k)_{ab} = \delta_{ab}` for :math:`a < d_k`, and the
full isometry is the tensor product :math:`E = \bigotimes_k E_k \in
\mathbb{C}^{D \times d}`. It satisfies :math:`E^\dagger E = I_d` and
:math:`P = E E^\dagger` is the projector onto the embedded computational subspace,
with complement :math:`P_\perp = I_D - P` projecting onto the leaked levels.

The behaviour of :func:`~quax.promote` depends on what is being promoted: states
are zero-padded, unitaries and operators are identity-extended, channels use a
**weighted** Kraus extension, and quantum instruments use an **incoherent**
extension. The sections below treat each in turn, then discuss the family of
superoperator extensions and why the weighted choice is the natural default for
the Kraus-trajectory simulations this library targets.


1. Promotion of state vectors
------------------------------

A pure state :math:`|\psi\rangle \in \mathbb{C}^d` is promoted by the isometry,

.. math::
   :label: eq-promote-sv

   |\tilde\psi\rangle = E\,|\psi\rangle \in \mathbb{C}^D .

Concretely this zero-pads the amplitude vector: the computational amplitudes are
preserved and the leaked levels are assigned zero amplitude. Because
:math:`E^\dagger E = I_d`, the norm is preserved, :math:`\langle\tilde\psi|
\tilde\psi\rangle = \langle\psi|\psi\rangle`, so a normalized state stays
normalized. The promoted state has no support on the complement,
:math:`P_\perp|\tilde\psi\rangle = 0`, which is exactly what we want for the
initial preparation of a freshly reset qubit: it starts entirely within the
computational subspace.

.. code-block:: python

   import quax as qx

   psi = qx.random_state_vector(dims=(2,))
   promoted = qx.promote(psi, (3,))   # amplitudes [a, b]  ->  [a, b, 0]


2. Promotion of density matrices
--------------------------------

A density matrix :math:`\rho` is promoted by conjugating with the isometry,

.. math::
   :label: eq-promote-dm

   \tilde\rho = E\,\rho\,E^\dagger .

This zero-pads the rows and columns that index the leaked levels: the
:math:`d \times d` computational block is preserved and every entry touching a
leaked level is zero. The map is consistent with promotion of pure states, since
:math:`E|\psi\rangle\langle\psi|E^\dagger = |\tilde\psi\rangle\langle\tilde\psi|`,
and it preserves the trace, :math:`\operatorname{Tr}\tilde\rho =
\operatorname{Tr}(E^\dagger E\,\rho) = \operatorname{Tr}\rho`. It is positive
semidefinite whenever :math:`\rho` is, so a valid state is promoted to a valid
state supported entirely on the computational subspace.

.. code-block:: python

   rho = qx.random_density_matrix(rank=2, dims=(2,))
   promoted = qx.promote(rho, (3,))   # 2x2 block preserved, leaked row/col are 0


3. Promotion of unitaries
-------------------------

States are *padded* with zeros, but a gate must leave any already-leaked
population untouched. A unitary :math:`U` (or a generic :class:`~quax.Operator`)
is therefore promoted by acting as :math:`U` on the computational subspace and as
the **identity** on the complement,

.. math::
   :label: eq-promote-unitary

   \tilde{U} = E\,U\,E^\dagger + P_\perp
   = \begin{pmatrix} U & 0 \\ 0 & I_{D-d} \end{pmatrix},

where the block form uses the basis ordering (computational levels first). The
result is unitary on the whole space: since :math:`E U E^\dagger` and
:math:`P_\perp` have orthogonal support, :math:`\tilde{U}^\dagger \tilde{U} =
E U^\dagger U E^\dagger + P_\perp = P + P_\perp = I_D`. A leaked basis state is an
eigenvector of eigenvalue one, :math:`\tilde{U}|{\perp}\rangle = |{\perp}\rangle`,
so the gate neither populates nor depopulates the leaked levels. Promotion of a
generic (non-unitary) :class:`~quax.Operator` follows the identical rule; only the
computational block changes.

.. code-block:: python

   u = qx.random_unitary(dims=((2,), (2,)))
   promoted = qx.promote(u, (3,))   # block-diag(U, 1): identity on |2>


4. Promotion of superoperators
------------------------------

Promoting a noise channel is the substantive case, because there is no longer a
*unique* faithful extension. Let :math:`\mathcal{E}` be a quantum channel on the
:math:`d`-dimensional system in Kraus form

.. math::
   :label: eq-promote-channel

   \mathcal{E}(\rho) = \sum_i K_i\, \rho\, K_i^\dagger,
   \qquad \sum_i K_i^\dagger K_i = I_d .

We want an embedded channel :math:`\tilde{\mathcal{E}}` on the
:math:`D`-dimensional space that (i) **reproduces** :math:`\mathcal{E}` on the
computational subspace and (ii) **acts as the identity** on the complement, since
a gate-level noise process should not by itself disturb a state that has already
leaked.

Embedding each Kraus operator by zero-padding, :math:`\hat{K}_i = E K_i E^\dagger`,
satisfies requirement (i) but **not** trace preservation:

.. math::
   :label: eq-padded-tp

   \sum_i \hat{K}_i^\dagger \hat{K}_i = E\Big(\sum_i K_i^\dagger K_i\Big)E^\dagger
   = E E^\dagger = P \ne I_D .

The padded operators annihilate the complement; we must restore the missing
:math:`P_\perp`. The key observation is that the complement projector can be
distributed across the Kraus operators *in any way we like*. Consider

.. math::
   :label: eq-family

   \tilde{K}_i = \hat{K}_i + \alpha_i P_\perp,
   \qquad \sum_i |\alpha_i|^2 = 1 .

Because :math:`\hat{K}_i` has support only on the computational subspace and
:math:`P_\perp` only on the complement, they have **disjoint support**:
:math:`\hat{K}_i^\dagger P_\perp = 0` and :math:`P_\perp \hat{K}_i = 0`. The cross
terms in :math:`\tilde{K}_i^\dagger \tilde{K}_i` vanish, and

.. math::
   :label: eq-family-tp

   \sum_i \tilde{K}_i^\dagger \tilde{K}_i
   = \sum_i \hat{K}_i^\dagger \hat{K}_i
   + \Big(\sum_i |\alpha_i|^2\Big) P_\perp
   = P + P_\perp = I_D .

So **every** weight assignment :math:`\{\alpha_i\}` with :math:`\sum_i |\alpha_i|^2
= 1` yields a completely positive, trace-preserving (CPTP) extension that meets
both requirements. The freedom in :math:`\{\alpha_i\}` is precisely the freedom in
*how the leaked subspace is correlated with the channel's Kraus (trajectory)
decomposition*. Different choices produce nearly identical average density
matrices but markedly different behaviour under a trajectory unravelling.

Quax resolves this freedom with the **weighted** extension (the default for
:class:`~quax.SuperOp`, :class:`~quax.KrausMap`, :class:`~quax.Choi`, and
:class:`~quax.PauliLiouville`), distributing the complement across every Kraus
operator in proportion to its Frobenius norm. The full family and the three
canonical choices are discussed in :ref:`Section 6 <promotion-choices>`; the
rationale for the weighted default is the subject of
:ref:`Section 7 <promotion-trajectories>`.

.. note::

   The weighting depends on the Kraus decomposition. A :class:`~quax.KrausMap` is
   promoted using *its own* operators, whereas :class:`~quax.SuperOp`,
   :class:`~quax.Choi`, and :class:`~quax.PauliLiouville` are first decomposed into
   the canonical Choi-eigenvector Kraus set. The resulting channels are CPTP and
   identical on the computational subspace, but their complement-coherence blocks
   (and hence their trajectory unravellings) reflect the chosen decomposition.


5. Promotion of quantum instruments
-----------------------------------

A :class:`~quax.QuantumInstrument` describes a measurement (or other
classically-conditioned process) as a collection of completely positive maps
:math:`\{\mathcal{I}_m\}`, one per classical outcome :math:`m`, whose sum
:math:`\sum_m \mathcal{I}_m` is trace preserving. Promoting an instrument is
different from promoting a channel: a measurement *destroys* coherence between the
computational and leaked subspaces, so the **incoherent** extension is the
physically correct one.

Each outcome map is zero-padded into the larger space,
:math:`\hat{\mathcal{I}}_m(\rho) = \sum_i \hat{K}_{m,i}\,\rho\,\hat{K}_{m,i}^\dagger`,
and the complement is restored by appending the projector :math:`P_\perp` as a
*separate* Kraus operator to a single outcome. Quax assigns it to the
highest-index outcome :math:`M`:

.. math::
   :label: eq-promote-instrument

   \tilde{\mathcal{I}}_m(\rho) =
   \begin{cases}
     \hat{\mathcal{I}}_m(\rho), & m < M, \\[4pt]
     \hat{\mathcal{I}}_M(\rho) + P_\perp\,\rho\,P_\perp, & m = M.
   \end{cases}

Summing over outcomes restores :math:`\sum_m \tilde{\mathcal{I}}_m` to a CPTP map
(the padded maps supply :math:`P` and the appended projector supplies
:math:`P_\perp`), while the separate :math:`P_\perp\,\rho\,P_\perp` term destroys
all coherence between the computational and complement subspaces. For dispersive
readout this matches the typical experimental behaviour: a leaked :math:`|2\rangle`
lands near the :math:`|1\rangle` IQ blob and is recorded as the highest
computational outcome, with no residual coherence to the measured subspace.


.. _promotion-choices:

6. Superoperator promotion choices
----------------------------------

Section 4 showed that every weight assignment :math:`\{\alpha_i\}` with
:math:`\sum_i |\alpha_i|^2 = 1` in Eq. :eq:`eq-family` gives a valid CPTP
extension. Three points in this family are of particular interest.

Coherent extension
~~~~~~~~~~~~~~~~~~~

Put the entire complement on a single Kraus operator, :math:`\alpha_0 = 1` and
:math:`\alpha_{i>0} = 0`:

.. math::
   :label: eq-coherent

   \tilde{K}_0 = \hat{K}_0 + P_\perp, \qquad \tilde{K}_{i>0} = \hat{K}_i .

This fully preserves coherence between the computational and complement subspaces
and is *exact* for a unitary channel — which has a single Kraus operator, so the
coherent extension reduces to Eq. :eq:`eq-promote-unitary`. Its drawback appears
in a trajectory simulation: the leaked state survives **only** along the
:math:`K_0` branch; every other branch (e.g. an :math:`X`, :math:`Y`, or
:math:`Z` error) annihilates it. The fate of the leaked population thereby becomes
spuriously correlated with which gate-error trajectory was sampled.

Incoherent extension
~~~~~~~~~~~~~~~~~~~~~

Add :math:`P_\perp` as a *separate* Kraus operator rather than folding it into an
existing one:

.. math::
   :label: eq-incoherent

   \tilde{\mathcal{E}}(\rho) = \sum_i \hat{K}_i\, \rho\, \hat{K}_i^\dagger
   + P_\perp\, \rho\, P_\perp .

This destroys all coherence between the computational and complement subspaces. It
is the physically correct extension for **measurements and resets**, where the
leaked levels genuinely decohere from the computational state, and is what Quax
uses when promoting a :class:`~quax.QuantumInstrument` (Section 5). It is exposed
directly as :func:`~quax.promote_incoherent`.

Weighted extension
~~~~~~~~~~~~~~~~~~~

Distribute the complement across *every* Kraus operator in proportion to its
Frobenius norm,

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
branches into Kraus operator :math:`i`. Because :math:`\tilde{K}_i|{\perp}\rangle
= \alpha_i |{\perp}\rangle` for any leaked basis state :math:`|{\perp}\rangle`, the
leaked state is preserved *identically in every branch*, while branch :math:`i`
fires with the channel's own intrinsic probability :math:`q_i`. The survival of a
leaked state is therefore **decoupled** from the computational gate trajectory.
This is the default extension used by :func:`~quax.promote` for channel types.


.. _promotion-trajectories:

7. Promoted superoperators in Kraus-trajectory simulations
----------------------------------------------------------

The motivation for the weighted default is the dominant use case for these
promoted channels: **Kraus-trajectory (Monte-Carlo wavefunction) simulation**. In
a trajectory unravelling, a pure state :math:`|\psi\rangle` is propagated through
the promoted channel by sampling a single Kraus branch :math:`i` with probability
:math:`p_i = \lVert \tilde{K}_i |\psi\rangle \rVert^2` and renormalizing. Averaging
over many trajectories reproduces the channel's action on the density matrix. The
choice of extension is invisible in the *average* but governs the *correlations*
along individual trajectories, and those correlations are exactly what a leakage
study measures.

The weighted extension is the natural method here for three reasons:

* **Trajectory decoupling.** A state that has leaked to :math:`|2\rangle` should
  propagate through the gate independently of the gate's own error channel. The
  weighted extension guarantees :math:`\tilde{K}_i|2\rangle = \alpha_i|2\rangle` in
  *every* branch, so the leaked population is never correlated with whether an
  :math:`I`, :math:`X`, :math:`Y`, or :math:`Z` error was sampled. The coherent
  extension (Eq. :eq:`eq-coherent`) fails this: it ties leakage survival to the
  :math:`K_0` branch alone.

* **No regression for gates.** A unitary has a single Kraus operator, so
  :math:`\alpha_0 = 1` and the weighted extension reduces to the coherent
  one — i.e. to the identity-extended unitary of Eq. :eq:`eq-promote-unitary`.
  Embedding ideal gates is therefore unchanged.

* **Correct averages.** Like every member of the family in Eq. :eq:`eq-family`,
  the weighted extension is CPTP and agrees with the other schemes on the
  computational and complement blocks of the output density matrix; the schemes
  differ only in the computational :math:`\leftrightarrow` complement coherence
  block.

Measurements and resets remain an exception: use :func:`~quax.promote_incoherent`
for those, as Quax does automatically when promoting a
:class:`~quax.QuantumInstrument`.

Worked example: depolarizing noise on a transmon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
along all four trajectories with the channel's intrinsic branch probabilities.

The schemes differ only in how the computational subspace stays coherent with the
leaked level. Apply the promoted channel to the coherent superposition
:math:`(|1\rangle + |2\rangle)/\sqrt{2}` and inspect the :math:`\rho_{12}` element:

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

* :func:`~quax.promote` — promote a quantum object: zero-pad states, identity-extend
  unitaries and operators, apply the **weighted** Kraus extension to channels, and
  apply the **incoherent** extension to quantum instruments.
* :func:`~quax.promote_incoherent` — the incoherent extension for measurements and
  resets, applied directly to a superoperator.
* :func:`~quax.embed` — positional embedding of an object into specified qudits of a
  larger register (built on :func:`~quax.promote`).
