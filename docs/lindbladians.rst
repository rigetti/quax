Lindbladian Generators
======================

Overview
--------

A **Lindbladian** (or Liouvillian) is the generator of a quantum dynamical semigroup,
describing open-system evolution under the Gorini–Kossakowski–Sudarshan–Lindblad (GKSL)
master equation:

.. math::

   \frac{d\rho}{dt} = \mathcal{L}[\rho]
   = -i[H, \rho] + \sum_k \Bigl( L_k \rho L_k^\dagger
     - \tfrac{1}{2}\{L_k^\dagger L_k,\, \rho\} \Bigr)

Here :math:`H` is the Hamiltonian and :math:`L_k` are **jump operators** that encode
dissipation.  Exponentiating the generator yields a **CPTP channel** for :math:`t \geq 0`:

.. math::

   \Phi(t) = e^{t \mathcal{L}}

This is the canonical way to generate physically valid (completely positive,
trace-preserving) open-system dynamics.

**Key distinction:** :class:`~quax.Lindbladian` is a *generator* — it is NOT a quantum
channel.  Use :func:`~quax.evolve` to convert it to a channel.


The Generator Algebra
---------------------

Lindbladians form a real vector space with several useful operations:

**Addition** — independent noise sources compose by adding their generators:

.. math::

   \mathcal{L}_1 + \mathcal{L}_2

This is *not* the same as combining jump operators before constructing the Lindbladian
(summing operators introduces cross-terms).  Always stack independent jump operators along
the ``n_ops`` axis or add Lindbladian objects.

**Scalar multiplication** — scale the overall noise rate:

.. math::

   \alpha \cdot \mathcal{L}

**Rate-scaling via** ``**`` — ``L ** alpha`` is equivalent to ``alpha * L``.  This scales
the generator rate, so ``evolve(L ** alpha, t) == evolve(L, alpha * t)``.

**Tensor product** ``|`` — compose independent subsystems A and B:

.. math::

   \mathcal{L}_{AB} = \mathcal{L}_A \otimes_\text{quax} \mathbf{I}_B
                    + \mathbf{I}_A \otimes_\text{quax} \mathcal{L}_B

Here :math:`\otimes_\text{quax}` denotes quax's superoperator tensor product, which embeds
each single-subsystem generator into the joint space (padding with the identity on the other
subsystem). Their sum is the *Kronecker sum* of the two generators — sometimes written
:math:`\mathcal{L}_A \oplus \mathcal{L}_B` — not an ordinary tensor product of the generators
themselves. It satisfies ``evolve(L_A | L_B, t) == evolve(L_A, t) | evolve(L_B, t)``.


The ``Lindbladian`` Object
--------------------------

.. autoclass:: quax.Lindbladian
   :members: from_matrix, from_operators, dims, matrix, num_ensemble_dims
   :undoc-members:

A :class:`~quax.Lindbladian` stores the :math:`d^2 \times d^2` generator matrix
:math:`\mathcal{L}` as a JAX tensor.  It supports ensemble dimensions (leading batch
axes) just like other quax objects.

**Tensor shape:** ``(*ensemble, d0_out_bra, …, d0_out_ket, …, d0_in_bra, …, d0_in_ket, …)``

**Matrix shape:** ``(*ensemble, d_out², d_in²)``


Constructing Lindbladians
-------------------------

Use :meth:`~quax.Lindbladian.from_operators` to build a Lindbladian from a Hamiltonian
and jump operators:

.. code-block:: python

   gen = qx.Lindbladian.from_operators(hamiltonian, jump_operators)

**Rate convention:** rates must be pre-absorbed into the jump operators.  Pass
``sqrt(γ) * L_physical`` rather than providing ``(L_physical, γ)`` separately.

**Stacking convention:** multiple independent jump operators must be stacked along a
leading ``n_ops`` axis — do **NOT** sum them first:

.. code-block:: python

   # Correct: stack jump operators
   L_stack = jnp.stack([L1, L2, L3])           # (3, d, d)
   jump_ops = qx.Operator.from_matrix(L_stack, (dims, dims))
   gen = qx.Lindbladian.from_operators(H, jump_ops)

   # Also correct: add separate Lindbladians
   gen = qx.Lindbladian.from_operators(H1, L1) + qx.Lindbladian.from_operators(H2, L2)

   # WRONG: summing jump operators before calling from_operators
   # (L1+L2)ρ(L1+L2)† ≠ L1ρL1† + L2ρL2†


Common Noise Channels
---------------------

All factories return a :class:`~quax.Lindbladian`.  Use :func:`~quax.evolve` to obtain
the corresponding CPTP channel.

+---------------------------------------+----------------------------------------------------------------+
| Factory                               | Jump operators                                                 |
+=======================================+================================================================+
| ``amplitude_damping_lindbladian(γ)``  | :math:`\sqrt{\gamma}\,|0\rangle\langle 1|`                     |
+---------------------------------------+----------------------------------------------------------------+
| ``dephasing_lindbladian(γ)``          | :math:`\sqrt{\gamma/2}\,Z`                                     |
+---------------------------------------+----------------------------------------------------------------+
| ``depolarizing_lindbladian(γ)``       | :math:`\sqrt{\gamma/3}\,\{X, Y, Z\}`                           |
+---------------------------------------+----------------------------------------------------------------+
| ``thermal_relaxation_lindbladian``    | :math:`\sqrt{1/T_1}\,\sigma_-`,                                |
| ``(t1, tphi)``                        | :math:`\sqrt{1/T_\varphi}/\sqrt{2}\,Z`                         |
+---------------------------------------+----------------------------------------------------------------+
| ``bit_flip_lindbladian(γ)``           | :math:`\sqrt{\gamma}\,X`                                       |
+---------------------------------------+----------------------------------------------------------------+
| ``phase_flip_lindbladian(γ)``         | :math:`\sqrt{\gamma}\,Z`                                       |
+---------------------------------------+----------------------------------------------------------------+
| ``leakage_lindbladian(γ)``            | :math:`\sqrt{\gamma}\,|2\rangle\langle 1|` (qutrit)            |
+---------------------------------------+----------------------------------------------------------------+
| ``seepage_lindbladian(γ)``            | :math:`\sqrt{\gamma}\,|1\rangle\langle 2|` (qutrit)            |
+---------------------------------------+----------------------------------------------------------------+

Qubit channel equivalences (at time *t*):

* ``amplitude_damping_lindbladian(γ)`` → ``relaxation_operators(1 − exp(−γt))``
* ``dephasing_lindbladian(γ)`` → ``dephasing_operators(1 − exp(−γt))``
* ``depolarizing_lindbladian(γ)`` → ``depolarizing_operators(¾(1 − exp(−4γt/3)))``
* ``bit_flip_lindbladian(γ)`` → ``bit_flip_operators(½(1 − exp(−2γt)))``
* ``phase_flip_lindbladian(γ)`` → ``phase_flip_operators(½(1 − exp(−2γt)))``


Evolving to a Channel
---------------------

:func:`~quax.evolve` dispatches on the input type:

* :class:`~quax.Observable` (Hamiltonian) → :class:`~quax.Unitary` via :math:`e^{-i t H}`
* :class:`~quax.Lindbladian` → :class:`~quax.SuperOp` via :math:`e^{t \mathcal{L}}`

.. code-block:: python

   channel = qx.evolve(gen, t)   # returns SuperOp, guaranteed CPTP for t >= 0

The evolution time ``t`` is a traceable JAX value — gradients flow through it:

.. code-block:: python

   grad_t = jax.grad(lambda t: loss(qx.evolve(gen, t)))(t0)


Promotion
---------

:func:`~quax.promote` embeds a Lindbladian in a larger Hilbert space by zero-padding
the generator tensor.  The higher-dimensional subspace receives a zero generator (trivial
evolution when no jump operators or Hamiltonian couple to it):

.. code-block:: python

   L_qutrit = qx.promote(L_qubit, (3,))   # returns Lindbladian with dims=((3,),(3,))

.. note::
   Zero-padding the generator is a mathematical embedding.  For states with coherences
   between the original and extended subspaces, the resulting channel may not be CPTP.
   To obtain a valid qutrit channel, construct the Lindbladian with qutrit-dimensioned
   jump operators from the start.


Complete Code Example
---------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import quax as qx

   # --- Combining independent noise sources ---
   # Thermal relaxation (T1=1µs, Tphi=2µs) plus small bit-flip noise
   L_thermal = qx.thermal_relaxation_lindbladian(t1=1.0, tphi=2.0)
   L_total = L_thermal + 0.01 * qx.bit_flip_lindbladian(gamma=1.0)

   # Evolve for 0.5µs → guaranteed CPTP
   channel = qx.evolve(L_total, t=0.5)
   print(qx.is_cptp(channel))   # True

   # --- Rate scaling ---
   # L ** alpha scales the noise rate: evolve(L**2, t) == evolve(L, 2*t)
   L_fast = L_total ** 2.0
   assert jnp.allclose(
       qx.evolve(L_fast, 0.5).matrix,
       qx.evolve(L_total, 1.0).matrix,
   )

   # --- Two-qubit independent noise ---
   L_A = qx.amplitude_damping_lindbladian(gamma=0.1)
   L_B = qx.dephasing_lindbladian(gamma=0.2)
   L_AB = L_A | L_B                    # Kronecker sum — 16×16 superoperator (acts on 4×4 density matrix)
   channel_AB = qx.evolve(L_AB, t=0.5)
   print(qx.is_cptp(channel_AB))       # True

   # Factorises as tensor product of single-qubit channels
   ch_A = qx.evolve(L_A, 0.5)
   ch_B = qx.evolve(L_B, 0.5)
   print(jnp.allclose(channel_AB.matrix, (ch_A | ch_B).matrix, atol=1e-8))  # True

   # --- Gradient through time ---
   target = qx.evolve(L_thermal, t=0.5)
   gen = qx.thermal_relaxation_lindbladian(t1=1.0, tphi=2.0)

   def loss(t):
       return jnp.real(jnp.sum(jnp.conj(target.matrix) * qx.evolve(gen, t).matrix))

   grad = jax.grad(loss)(jnp.array(0.4))
   print(jnp.isfinite(grad))   # True

   # --- Leakage + seepage on a qutrit ---
   L_leak = qx.leakage_lindbladian(gamma=0.05)
   L_seep = qx.seepage_lindbladian(gamma=0.02)
   L_qutrit = L_leak + L_seep
   channel_qutrit = qx.evolve(L_qutrit, t=1.0)
   print(qx.is_cptp(channel_qutrit))   # True
