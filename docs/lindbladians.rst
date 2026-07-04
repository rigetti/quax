Lindbladian Generators
======================

Overview
--------

A **Lindbladian** (or Liouvillian) is the generator of a quantum dynamical semigroup,
describing open-system evolution under the
`Gorini–Kossakowski–Sudarshan–Lindblad (GKSL) <https://en.wikipedia.org/wiki/Lindbladian>`_
master equation:

.. math::

   \frac{d\rho}{dt} = \mathcal{L}[\rho]
   = -i[H, \rho] + \sum_k \Bigl( L_k \rho L_k^\dagger
     - \tfrac{1}{2}\{L_k^\dagger L_k,\, \rho\} \Bigr)

Here :math:`H` is the Hamiltonian generating coherent evolution, and the :math:`L_k` are
**jump operators** (also called Lindblad or collapse operators).  Each :math:`L_k` models an
irreversible dissipative process — energy decay, dephasing, leakage, etc. — through which the
system couples to its environment: the term :math:`L_k \rho L_k^\dagger` transfers population
and coherence *along* the jump, while the anticommutator
:math:`-\tfrac12\{L_k^\dagger L_k,\, \rho\}` supplies the matching no-jump damping.
Exponentiating the generator yields a **CPTP channel** for :math:`t \geq 0`:

.. math::

   \Phi(t) = e^{t \mathcal{L}}

This is the canonical way to generate physically valid (completely positive,
trace-preserving) open-system dynamics.

**Key distinction:** A :class:`~quax.Lindbladian` is a *generator*, not a quantum channel.
Like a :class:`~quax.SuperOp` it is stored as a :math:`d^2 \times d^2` matrix acting on a
vectorized density matrix, but it represents the instantaneous rate of change
:math:`\dot\rho = \mathcal{L}[\rho]` — not a finite-time map.  Only its exponential
:math:`e^{t\mathcal{L}}` is a CPTP superoperator (an actual channel).  Use
:func:`~quax.evolve` to exponentiate a Lindbladian into a :class:`~quax.SuperOp`.


The Generator Algebra
---------------------

Lindbladians form a real vector space with several useful operations:

**Addition** — independent noise sources compose by adding their generators:

.. math::

   \mathcal{L}_1 + \mathcal{L}_2

This is *not* the same as combining jump operators before constructing the Lindbladian.
Summing them first introduces spurious cross-terms in the dissipator:

.. math::

   (L_1 + L_2)\,\rho\,(L_1 + L_2)^\dagger
   = L_1 \rho L_1^\dagger + L_2 \rho L_2^\dagger
     + \underbrace{L_1 \rho L_2^\dagger + L_2 \rho L_1^\dagger}_{\text{unphysical cross-terms}}

so :math:`(L_1 + L_2)` describes a *different* physical process than two independent jump
channels.  Always stack independent jump operators along the ``n_ops`` axis or add Lindbladian
objects.

**Scalar multiplication** — scale the overall noise rate:

.. math::

   \alpha \cdot \mathcal{L}

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

A :class:`~quax.Lindbladian` stores its Hamiltonian (an :class:`~quax.Observable`, or ``None``)
and jump operators (an :class:`~quax.Operator`) directly; the :math:`d^2 \times d^2` generator
matrix :math:`\mathcal{L}` is computed on demand from them and cached, and exposed via the
``.matrix`` attribute.  It supports ensemble dimensions (leading batch axes) just like other quax
objects (the jump operators' ``n_ops`` axis is *not* an ensemble axis).

**Matrix shape:** ``(*ensemble, d_out², d_in²)`` (via ``.matrix``)


Constructing Lindbladians
-------------------------

Construct a :class:`~quax.Lindbladian` directly from a Hamiltonian and jump operators:

.. code-block:: python

   gen = qx.Lindbladian(hamiltonian=hamiltonian, jump_operators=jump_operators)

Pass ``hamiltonian=None`` for a purely dissipative generator.

**Rate convention:** rates must be pre-absorbed into the jump operators.  Pass
``sqrt(γ) * L_physical`` rather than providing ``(L_physical, γ)`` separately.

**Stacking convention:** multiple independent jump operators must be stacked along a
leading ``n_ops`` axis — do **NOT** sum them first:

.. code-block:: python

   # Correct: stack jump operators
   L_stack = jnp.stack([L1, L2, L3])           # (3, d, d)
   jump_ops = qx.Operator.from_matrix(L_stack, (dims, dims))
   gen = qx.Lindbladian(hamiltonian=H, jump_operators=jump_ops)

   # Also correct: add separate Lindbladians
   gen = qx.Lindbladian(hamiltonian=H1, jump_operators=L1) + qx.Lindbladian(hamiltonian=H2, jump_operators=L2)

   # WRONG: summing jump operators before construction
   # (L1+L2)ρ(L1+L2)† ≠ L1ρL1† + L2ρL2†


Inspecting Operators
--------------------

A :class:`~quax.Lindbladian` stores its Hamiltonian and jump operators directly (they are its
only data), so they are always recoverable — exactly as supplied — via the ``.hamiltonian`` and
``.jump_operators`` attributes (``hamiltonian`` is ``None`` for a purely dissipative generator):

.. code-block:: python

   H, jump_ops = gen.hamiltonian, gen.jump_operators

.. note::
   There is **no gauge canonicalization**: the physical operators are stored verbatim (a single
   jump operator stays a single jump operator).  Because the operators are the source of truth,
   only physically valid (CPTP-generating) Lindbladians are representable — operations that could
   produce a non-CP generator (negation, subtraction, and multiplication by a negative or complex
   scalar) raise instead of silently returning an invalid object.


Common Noise Channels
---------------------

The factories live in the :mod:`quax.lindbladians` submodule and each returns a
:class:`~quax.Lindbladian`.  Use :func:`~quax.evolve` to obtain the corresponding CPTP
channel.  See each factory's docstring in the API reference for its jump operators.  These
generators are the canonical way to build the common noise channels; evolving one for time
*t* gives, e.g. for amplitude damping, the channel with damping probability
:math:`1 - e^{-\gamma t}` (and analogously ``p = ¾(1 − e^{-4γt/3})`` for depolarizing).


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

:func:`~quax.promote` embeds a Lindbladian in a larger Hilbert space at the **operator
level**: it zero-pads the stored Hamiltonian and jump operators into the larger space (the
operator-level :func:`~quax.promote`) and rebuilds the generator via the GKSL formula:

.. code-block:: python

   L_qutrit = qx.promote(L_qubit, (3,))   # returns Lindbladian with dims=((3,),(3,))

The result is a valid generator whose exponential is CPTP for :math:`t \geq 0`.  It agrees
with the qubit generator on the computational subspace and **correctly damps** coherences
between the original and added levels — e.g. amplitude damping ``L = √γ|0⟩⟨1|`` has
``L†L = γ|1⟩⟨1|``, whose ``-½{L†L, ρ}`` term decays the ``ρ₁₂``/``ρ₂₁`` coherences at rate
γ/2.

.. note::
   Promotion is *not* a naive zero-padding of the generator matrix.  Zero-padding the
   generator would freeze the new cross-subspace coherences while population still decays,
   which is not a valid GKSL generator — its exponential is not completely positive.  Embedding
   the stored jump operators and rebuilding the generator avoids this and coincides with building
   the Lindbladian natively in the larger space.


Complete Code Example
---------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import quax as qx

   # --- Combining independent noise sources ---
   # Thermal relaxation (T1=1µs, Tphi=2µs) plus small bit-flip noise
   L_thermal = qx.lindbladians.thermal_relaxation(t1=1.0, tphi=2.0)
   L_total = L_thermal + 0.01 * qx.lindbladians.bit_flip(gamma=1.0)

   # Evolve for 0.5µs → guaranteed CPTP
   channel = qx.evolve(L_total, t=0.5)
   print(qx.is_cptp(channel))   # True

   # --- Rate scaling ---
   # Scaling the generator scales the noise rate: evolve(2*L, t) == evolve(L, 2*t)
   L_fast = 2.0 * L_total
   assert jnp.allclose(
       qx.evolve(L_fast, 0.5).matrix,
       qx.evolve(L_total, 1.0).matrix,
   )

   # --- Two-qubit independent noise ---
   L_A = qx.lindbladians.amplitude_damping(gamma=0.1)
   L_B = qx.lindbladians.dephasing(gamma=0.2)
   L_AB = L_A | L_B                    # Kronecker sum — 16×16 superoperator (acts on 4×4 density matrix)
   channel_AB = qx.evolve(L_AB, t=0.5)
   print(qx.is_cptp(channel_AB))       # True

   # Factorises as tensor product of single-qubit channels
   ch_A = qx.evolve(L_A, 0.5)
   ch_B = qx.evolve(L_B, 0.5)
   print(jnp.allclose(channel_AB.matrix, (ch_A | ch_B).matrix, atol=1e-8))  # True

   # --- Gradient through time ---
   target = qx.evolve(L_thermal, t=0.5)
   gen = qx.lindbladians.thermal_relaxation(t1=1.0, tphi=2.0)

   def loss(t):
       return jnp.real(jnp.sum(jnp.conj(target.matrix) * qx.evolve(gen, t).matrix))

   grad = jax.grad(loss)(jnp.array(0.4))
   print(jnp.isfinite(grad))   # True

   # --- Leakage + seepage on a qutrit ---
   L_leak = qx.lindbladians.leakage(gamma=0.05)
   L_seep = qx.lindbladians.seepage(gamma=0.02)
   L_qutrit = L_leak + L_seep
   channel_qutrit = qx.evolve(L_qutrit, t=1.0)
   print(qx.is_cptp(channel_qutrit))   # True
