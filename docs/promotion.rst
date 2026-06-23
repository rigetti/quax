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
and :math:`D = \prod_k D_k` for the total dimensions, and let :math:`E` be the
embedding isometry that injects the computational space into the promoted one,
defined in the note below.

.. note::

   **Embedding isometry.** For each subsystem the embedding isometry
   :math:`E_k \in \mathbb{C}^{D_k \times d_k}` is the zero-padded identity,
   :math:`(E_k)_{ab} = \delta_{ab}` for :math:`a < d_k` (it copies the
   computational basis states :math:`|0\rangle, \dots, |d_k - 1\rangle` into the
   first :math:`d_k` levels of the larger space and sends nothing to the leaked
   levels). The full isometry is the tensor product
   :math:`E = \bigotimes_k E_k \in \mathbb{C}^{D \times d}`. It satisfies
   :math:`E^\dagger E = I_d` (it preserves inner products, so it is an isometry but
   not unitary), while :math:`P = E E^\dagger` is the projector onto the embedded
   computational subspace and :math:`P_\perp = I_D - P` projects onto the leaked
   levels.

The behaviour of :func:`~quax.promote` depends on what is being promoted: states
and operators are zero-padded, unitaries are identity-extended, channels use the
**coherent** Kraus extension, and quantum instruments use an **incoherent**
extension. The sections below treat each in turn, then discuss the family of
superoperator extensions, the Lindbladian justification for the coherent default,
and the important caveats that every user of channel promotion must understand.


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
population untouched. A unitary :math:`U` is therefore promoted by acting as
:math:`U` on the computational subspace and as the **identity** on the complement,

.. math::
   :label: eq-promote-unitary

   \tilde{U} = E\,U\,E^\dagger + P_\perp
   = \begin{pmatrix} U & 0 \\ 0 & I_{D-d} \end{pmatrix},

where the block form uses the basis ordering (computational levels first). The
result is unitary on the whole space: since :math:`E U E^\dagger` and
:math:`P_\perp` have orthogonal support, :math:`\tilde{U}^\dagger \tilde{U} =
E U^\dagger U E^\dagger + P_\perp = P + P_\perp = I_D`. A leaked basis state is an
eigenvector of eigenvalue one, :math:`\tilde{U}|{\perp}\rangle = |{\perp}\rangle`,
so the gate neither populates nor depopulates the leaked levels.

.. code-block:: python

   u = qx.random_unitary(dims=((2,), (2,)))
   promoted = qx.promote(u, (3,))   # block-diag(U, 1): identity on |2>


4. Promotion of operators
--------------------------

A generic :class:`~quax.Operator` (which may be non-unitary or non-Hermitian) is
**zero-padded** into the larger space:

.. math::
   :label: eq-promote-operator

   \tilde{A} = E\,A\,E^\dagger
   = \begin{pmatrix} A & 0 \\ 0 & 0 \end{pmatrix}.

Unlike a unitary, a general operator carries no constraint that forces the
complement subspace to be left unchanged. Zero-padding — embedding :math:`A` as
the top-left block of a larger zero matrix — is the natural linear extension: it
preserves the operator's action on the computational subspace and assigns no
output to states in the complement.

.. code-block:: python

   op = qx.random_operator(dims=((2,), (2,)), key=jax.random.key(0))
   promoted = qx.promote(op, (3,))   # 2×2 block preserved, leaked row/col are 0


5. Promotion of quantum instruments
-----------------------------------

A :class:`~quax.QuantumInstrument` describes a measurement (or other
classically-conditioned process) as a collection of completely positive maps
:math:`\{\mathcal{I}_m\}`, one per classical outcome :math:`m`, whose sum
:math:`\sum_m \mathcal{I}_m` is trace preserving. A measurement *destroys*
coherence between the computational and leaked subspaces, so an instrument is
promoted by the **incoherent** extension (the channel case, where that coherence
is instead retained, is treated in Section 6).

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

Why the incoherent extension here rather than the coherent one used for channels?
The distinction is physical, not merely conventional. A measurement extracts
classical information: recording an outcome collapses the state and severs any
coherence between the subspace that was "read out" and the leaked levels, exactly
the decoherence that the separate :math:`P_\perp\,\rho\,P_\perp` term enforces.


6. Promotion of superoperators
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

Why can we not simply mirror the unitary embedding of Eq. :eq:`eq-promote-unitary`?
For a unitary the prescription "act as :math:`U` inside, as the identity outside"
is unambiguous because a unitary is a *single* operator with a single,
well-defined action on the complement. A channel is instead an operator *sum*, and
the trace-preservation constraint :math:`\sum_i K_i^\dagger K_i = I_d` couples the
Kraus operators together — there is no single object to identity-extend. The
naive superoperator analogue, conjugating the channel by the embedding isometry
(:math:`\tilde{\mathcal{E}} = E \mathcal{E} E^\dagger` at the map level), runs into
exactly the obstruction below: it annihilates the complement and is not trace
preserving. This forces a *choice* of how the restored complement
correlates with the channel's action — coherently, incoherently, or somewhere in
between — and that choice lives in the operator-sum representation. We therefore
work with Kraus operators, where the freedom is explicit and the CPTP constraint
is easy to enforce. We note that this choice is fundamentally ambiguous for
a superoperator in isolation. A more rigorous approach is to specify the subspace
coherences exactly, which can be done if the channel arises from a known
Lindbladian generator.

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
decomposition*. Different choices produce identical average density matrices in the
computational and complement blocks, but differ in the cross-block
(computational :math:`\leftrightarrow` complement) coherences.

To resolve this freedom, Quax chooses the **coherent** extension (the default for
:class:`~quax.SuperOp`, :class:`~quax.KrausMap`, :class:`~quax.Choi`, and
:class:`~quax.PauliLiouville`), which places the entire complement weight on the
first Kraus operator :math:`K_0`. This is fundamentally an arbitrary choice, as
the subspace coherences are not specified by the superoperator alone. However,
for many physically relevant channels, the coherent extension gives intuitive
results. The physical justification for this choice and its important limitations
are discussed in detail under
:ref:`Promotion choices <promotion-choices>` below.


.. _promotion-choices:

Promotion choices
~~~~~~~~~~~~~~~~~~

The family in Eq. :eq:`eq-family` shows that every weight assignment
:math:`\{\alpha_i\}` with :math:`\sum_i |\alpha_i|^2 = 1` gives a valid CPTP
extension. Three points in this family are of particular interest.

Coherent extension (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Put the entire complement on a single Kraus operator, :math:`\alpha_0 = 1` and
:math:`\alpha_{i>0} = 0`:

.. math::
   :label: eq-coherent

   \tilde{K}_0 = \hat{K}_0 + P_\perp, \qquad \tilde{K}_{i>0} = \hat{K}_i .

This is the **default extension** used by :func:`~quax.promote` for channel types. 
It fully preserves coherence between the computational and complement subspaces 
and is *exact* for a unitary channel — which has a single Kraus operator, so the 
coherent extension reduces exactly to Eq. :eq:`eq-promote-unitary`.

**The Lindbladian justification.** The most physically transparent way to promote
a channel is to work at the level of the underlying microscopic model rather than
the channel itself. Suppose the qubit channel arises from a Lindbladian master
equation with Hamiltonian :math:`H` and jump operators :math:`\{L_k\}` all
supported within the :math:`d`-dimensional computational subspace, so the
:math:`D`-dimensional promoted dynamics are obtained simply by zero-padding every
operator:

.. math::
   :label: eq-lindblad-promote

   \hat{H} = E H E^\dagger, \qquad \hat{L}_k = E L_k E^\dagger .

Because the zero-padded operators satisfy :math:`\hat{H} P_\perp = \hat{L}_k
P_\perp = 0` (a consequence of :math:`E^\dagger P_\perp = 0`), the Lindbladian
generator has **zero** action on any state supported entirely in the complement:
:math:`\hat{\mathcal{L}}(P_\perp\rho P_\perp) = 0`. Complement states are
therefore stationary — their time derivative is identically zero. Exponentiating a
zero generator gives the identity (not zero) in the complement block, so the
promoted channel :math:`\hat{\mathcal{E}}_t` leaves the complement exactly
untouched while reproducing the qubit channel on the computational subspace. For many
physically relevant channels — amplitude damping, cascade decay, thermal
relaxation — this zero-padded Lindbladian approach agrees *exactly* with the
coherent extension. The agreement holds whenever the first Kraus operator
:math:`K_0` from the canonical (Choi-eigenvector) decomposition equals the no-jump
propagator :math:`\exp\!\bigl(-\tfrac{1}{2}\sum_k L_k^\dagger L_k\, t\bigr)`.
This can be verified analytically for a broad class of channels and is confirmed by
the tests in ``tests/test_promotion.py``.

**The depolarizing argument.** The coherent extension also has a direct physical
interpretation in terms of Kraus trajectories. Consider depolarizing noise on the
computational subspace. In a trajectory unravelling, a jump event corresponds to
one of the Pauli error operators :math:`X`, :math:`Y`, or :math:`Z` firing. These
operators are zero-padded and therefore have *zero amplitude on any leaked state*:
:math:`\hat{K}_X |2\rangle = \hat{K}_Y |2\rangle = \hat{K}_Z |2\rangle = 0`. So
observing a jump event in the computational-subspace error channel necessarily
implies the system was *not* in the leaked level at the time of the jump — and
therefore the post-jump state has **zero probability of being in** :math:`|2\rangle`.
This is the physically correct conclusion: a depolarizing interaction that acts
entirely within the qubit subspace only happens *to* a qubit, so a subsequent
measurement of whether the qubit is leaked must give "no." The coherent extension
encodes exactly this logic. Only the no-error branch (:math:`K_0`) carries the
complement projector and therefore preserves any pre-existing leaked population,
which is also correct: a system that was leaked and experienced no jump simply
continues to be leaked.

.. _promotion-decomposition-warning:

.. warning::

   **The coherent extension is Kraus-decomposition-dependent. Use it with care.**

   Unlike the incoherent extension, the coherent extension gives a promoted channel
   that depends on *which* Kraus operator is called :math:`K_0`. Two Kraus
   representations of the **same** channel can produce **different** promoted
   channels. The promoted channels have identical action on the computational
   subspace and on the complement subspace in isolation; they differ only in the
   cross-block (computational :math:`\leftrightarrow` complement) coherences. This
   difference is invisible in average density-matrix simulations but shows up in
   trajectory unravellings and in any observable that mixes the two subspaces.

   **Phase of** :math:`K_0` **also matters.** Because :math:`P_\perp` is a real
   projector and does not share the global phase of :math:`K_0`, replacing
   :math:`K_0` by :math:`e^{i\theta} K_0` changes the promoted channel. Quax
   canonicalizes the phase by requiring that the element of :math:`K_0` with the
   largest magnitude be real and positive. This convention is applied uniformly —
   including when inputs are batched — so that results are independent of batch
   size. Any external code that constructs Kraus operators and feeds them as a
   :class:`~quax.KrausMap` should be aware that the global phase of the first
   operator affects the promoted channel.

   **Which decomposition does Quax use?** When you pass a :class:`~quax.SuperOp`,
   :class:`~quax.Choi`, or :class:`~quax.PauliLiouville` to :func:`~quax.promote`,
   Quax internally converts to the **canonical Choi-eigenvector decomposition**:
   :math:`K_0` is the Kraus operator corresponding to the *largest* eigenvalue of
   the Choi matrix. This is unique (up to the global-phase convention above) and
   corresponds physically to the dominant no-error branch. When you pass a
   :class:`~quax.KrausMap` directly, Quax honours **your** :math:`K_0` (the first
   operator in the map) — so the two paths can yield different promoted channels for
   the same underlying channel.

   **Practical guidance.** If the channel arises from a known Lindbladian with
   clearly identified jump operators, the most faithful promotion is via the
   Lindbladian zero-padding route described above: construct the promoted channel
   by zero-padding :math:`H` and each :math:`L_k`, then exponentiate. The coherent
   extension of the canonical Choi-eigenvector decomposition agrees with this for
   many common channels but should be verified on a case-by-case basis. If in doubt,
   supply the Lindblad-derived channel directly rather than promoting a pre-computed
   superoperator.


Incoherent extension
^^^^^^^^^^^^^^^^^^^^

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
directly as :func:`~quax.promote_incoherent`. The incoherent extension is
decomposition-independent: the :math:`P_\perp\,\rho\,P_\perp` contribution is the
same regardless of which Kraus set represents :math:`\mathcal{E}`.



.. _promotion-trajectories:

Use in Kraus-trajectory simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The choice of extension matters most in **Kraus-trajectory (Monte-Carlo
wavefunction) simulation**, where a pure state :math:`|\psi\rangle` is propagated
by sampling a single Kraus branch :math:`i` with probability
:math:`p_i = \lVert \tilde{K}_i |\psi\rangle \rVert^2` and renormalizing.
Averaging over many trajectories reproduces the channel's action on the density
matrix. The extension choice is invisible in the *average* but governs the
*correlations* along individual trajectories.

Under the **coherent extension** (the default):

* **Error branches annihilate leaked population.** For noise whose jump operators
  are zero-padded within the computational subspace (e.g. depolarizing, amplitude
  damping), each error Kraus :math:`\hat{K}_{i>0}` has no matrix elements for
  leaked states. Sampling an error branch therefore collapses the leaked probability
  to zero, which is the physically correct outcome: observing a qubit-subspace
  interaction implies the system was in the qubit subspace.
* **The no-error branch preserves leaked population.** :math:`\tilde{K}_0 =
  \hat{K}_0 + P_\perp` maps any leaked state to itself. A system that was leaked
  and experienced no jump simply remains leaked, with a norm factor from
  :math:`\hat{K}_0`.
* **Exact for unitaries.** A unitary channel has a single Kraus operator, so
  :math:`K_0 = U` and the coherent extension reduces to
  Eq. :eq:`eq-promote-unitary`. Promoting ideal gates is unchanged.

Measurements and resets use the **incoherent extension** regardless of this
setting, as Quax does automatically when promoting a
:class:`~quax.QuantumInstrument`.

Worked example: depolarizing noise on a transmon
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider single-qubit depolarizing noise,

.. math::
   :label: eq-depol

   \mathcal{E}(\rho) = (1-p)\,\rho + \frac{p}{3}\big(X\rho X + Y\rho Y + Z\rho Z\big),

with canonical Choi-eigenvector Kraus operators :math:`K_0 = \sqrt{1-p}\,I` and
:math:`K_{1,2,3} = \sqrt{p/3}\,\{Z, X, Y\}`. Promoting to a qutrit
(:math:`d = 2 \to D = 3`) with :math:`P_\perp = |2\rangle\!\langle 2|` via the
coherent extension gives

.. math::
   :label: eq-depol-promoted

   \tilde{K}_0 = \begin{pmatrix} \sqrt{1-p}\,I & 0 \\ 0 & 1 \end{pmatrix}, \qquad
   \tilde{K}_{i>0} = \begin{pmatrix} K_i & 0 \\ 0 & 0 \end{pmatrix}.

The no-error branch preserves :math:`|2\rangle`, while the error branches
(:math:`Z`, :math:`X`, :math:`Y`) annihilate it. In Quax:

.. code-block:: python

   import jax.numpy as jnp
   import quax as qx

   p = 0.1
   channel = qx.depolarizing_operators(p)    # qubit KrausMap (d = 2)
   promoted = qx.promote(channel, (3,))      # coherent extension (default)

   # The no-error branch (K_0) maps |2> to |2> with amplitude 1.
   # Every error branch (K_1, K_2, K_3) maps |2> to 0.
   print(promoted.matrix[:, 2, 2])           # [1.0, 0.0, 0.0, 0.0]

The difference between the two extensions is most directly visible in the
Weyl-Liouville (generalized Pauli transfer) matrix of the promoted channel.
The 9×9 Weyl-Liouville matrix of a qutrit superoperator has rows and columns
labelled by the Weyl–Heisenberg operator basis :math:`\{W_{xz}\}` of the
three-level system. The :math:`(W_{00}, W_{10}, W_{01}, W_{11})` corner recovers
the standard qubit Pauli-transfer matrix and is identical for both extensions.
The rest of the matrix reveals how the two choices treat the leaked subspace and
its coherences with the computational levels.

.. figure:: _static/promotion-weyl-coherent.png
   :align: center
   :width: 90%

   **Coherent extension** of 1-qubit depolarizing (:math:`p = 0.1`) promoted to a
   qutrit. The matrix is nearly diagonal throughout. The top-left
   :math:`3 \times 3` block (:math:`W_{00}`, :math:`W_{01}`, :math:`W_{02}` —
   the population/diagonal operators) is completely decoupled from the remaining
   :math:`6 \times 6` coherence block. All six coherence operators (:math:`W_{11}`,
   :math:`W_{22}`, :math:`W_{10}`, :math:`W_{20}`, :math:`W_{21}`,
   :math:`W_{12}`) decay by the same uniform factor :math:`{\approx}0.92`, with
   only small numerical off-diagonal entries. This clean structure reflects that
   the coherent extension treats the qutrit coherences uniformly — each coherence
   channel evolves independently.

.. figure:: _static/promotion-weyl-incoherent.png
   :align: center
   :width: 90%

   **Incoherent extension** of the same channel. The top-left :math:`3 \times 3`
   population block is identical to the coherent case — the two extensions agree
   on all population dynamics. The :math:`6 \times 6` coherence block is
   dramatically different: the diagonal entries fall to :math:`{\approx}0.29` and
   large off-diagonal entries (up to :math:`{\pm}0.25`) appear throughout. This
   dense structure arises because projecting out the
   :math:`|0\rangle\!\langle 2|` and :math:`|1\rangle\!\langle 2|` coherences
   (as the incoherent :math:`P_\perp\rho P_\perp` term does) strongly mixes the
   Weyl coherence operators, coupling channels that the coherent extension keeps
   separate.

Starting from a coherent superposition :math:`(|1\rangle + |2\rangle)/\sqrt{2}`,
the depolarizing channel decoheres the qubit component while the no-error branch
maintains cross-subspace coherence:

.. code-block:: python

   psi = jnp.array([0.0, 1.0, 1.0], dtype=complex) / jnp.sqrt(2.0)
   rho = jnp.outer(psi, psi.conj())

   coherent = qx.kraus_to_superop(qx.promote(channel, (3,)))
   incoherent = qx.promote_incoherent(qx.kraus_to_superop(channel), (3,))

   rho_c = (coherent.matrix  @ rho.ravel()).reshape(3, 3)
   rho_i = (incoherent.matrix @ rho.ravel()).reshape(3, 3)

   print(abs(rho_c[1, 2]))   # > 0  : coherent retains computational<->leaked coherence
   print(abs(rho_i[1, 2]))   # = 0  : incoherent destroys it

The coherent (default) extension retains a partial :math:`\rho_{12}` coherence
via the no-error Kraus branch, whereas the incoherent extension — appropriate for
a measurement or reset — removes it entirely.


API reference
-------------

* :func:`~quax.promote` — promote a quantum object: zero-pad states and operators,
  identity-extend unitaries, apply the **coherent** Kraus extension to channels, and
  apply the **incoherent** extension to quantum instruments.
* :func:`~quax.promote_incoherent` — the incoherent extension for measurements and
  resets, applied directly to a superoperator.
* :func:`~quax.embed` — positional embedding of an object into specified qudits of a
  larger register (built on :func:`~quax.promote`).
