Quantum Objects
===============

Quax provides a hierarchy of quantum object types for representing states, operators, and
superoperators.  Every type inherits from :class:`~quax.QuantumObject` and stores its data
as a JAX tensor whose trailing axes encode the quantum degrees of freedom, together with a
``num_qubits`` integer that records how many qudits are described.

.. contents:: On this page
   :local:
   :depth: 2

Type Hierarchy
--------------

.. code-block:: text

   QuantumObject
   ├── State
   │   ├── StateVector          |ψ⟩  — pure state vector
   │   └── DensityMatrix        ρ    — mixed state
   ├── Operator                 O    — general linear operator
   │   ├── Observable           A    — A = A† (Hermitian)
   │   └── Unitary              U    — U U† = I
   │       └── Involution       A    — A = A† and A² = I  (e.g. Paulis, Hadamard)
   └── SuperOperator
       ├── SuperOp              S    — Liouville / superoperator matrix
       ├── KrausMap             K    — {Kᵢ} (Kraus operators)
       ├── Choi                 J    — Choi–Jamiołkowski matrix
       ├── Chi                  χ    — process / χ matrix  (limited support)
       └── PauliLiouville       P    — Pauli transfer matrix

Type Abbreviations
------------------

The operator tables below use the following short names.

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Symbol
     - Type
     - Description
   * - **SV**
     - :class:`~quax.StateVector`
     - Pure state vector |ψ⟩
   * - **DM**
     - :class:`~quax.DensityMatrix`
     - Mixed state density matrix ρ
   * - **U**
     - :class:`~quax.Unitary`
     - Unitary operator
   * - **Op**
     - :class:`~quax.Operator`
     - General linear operator
   * - **Obs**
     - :class:`~quax.Observable`
     - Hermitian operator A = A†
   * - **Inv**
     - :class:`~quax.Involution`
     - Involutory unitary A = A† and A² = I
   * - **S**
     - :class:`~quax.SuperOp`
     - Superoperator (Liouville) matrix
   * - **K**
     - :class:`~quax.KrausMap`
     - Kraus channel
   * - **J**
     - :class:`~quax.Choi`
     - Choi matrix
   * - **χ**
     - :class:`~quax.Chi`
     - Chi (process) matrix
   * - **P**
     - :class:`~quax.PauliLiouville`
     - Pauli–Liouville (transfer) matrix

Binary Operator Tables
----------------------

The cells show the **return type** of each operation.
**—** indicates the operation is not defined between those types (a :exc:`TypeError` is
raised at runtime).
**NI** indicates the method exists but raises :exc:`NotImplementedError` (the
:class:`~quax.Chi` type has limited implementation support).

Addition (``a + b``)
~~~~~~~~~~~~~~~~~~~~

Addition is defined only for :class:`~quax.Operator` subtypes.  The sum of CPTP maps is
not generally CPTP, so ``+`` is intentionally absent from all :class:`~quax.SuperOperator`
subtypes.

Preservation rules:

- **Observable + Observable → Observable** (sum of Hermitian matrices is Hermitian).
- Any other mix involving a non-Hermitian :class:`~quax.Operator` or :class:`~quax.Unitary`
  downgrades the result to **Operator**.
- :class:`~quax.Involution` inherits :class:`~quax.Observable`'s addition rules; the
  sum of two involutions is *Observable*, not *Involution* (sum is not guaranteed to
  satisfy A² = I).

.. csv-table::
   :header: "+",      SV, DM, U,  Op, Obs, Inv, S,  K,  J,  χ,  P
   :widths:  14, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
   :stub-columns: 1

   **SV**,    —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **DM**,    —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **U**,     —,  —,  Op, Op, Op,  Op,  —,  —,  —,  —,  —
   **Op**,    —,  —,  Op, Op, Op,  Op,  —,  —,  —,  —,  —
   **Obs**,   —,  —,  Op, Op, Obs, Obs, —,  —,  —,  —,  —
   **Inv**,   —,  —,  Op, Op, Obs, Obs, —,  —,  —,  —,  —
   **S**,     —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **K**,     —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **J**,     —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **χ**,     —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **P**,     —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —

Subtraction (``a - b``)
~~~~~~~~~~~~~~~~~~~~~~~~

Subtraction follows identical rules to addition.

.. csv-table::
   :header: "-",      SV, DM, U,  Op, Obs, Inv, S,  K,  J,  χ,  P
   :widths:  14, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
   :stub-columns: 1

   **SV**,    —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **DM**,    —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **U**,     —,  —,  Op, Op, Op,  Op,  —,  —,  —,  —,  —
   **Op**,    —,  —,  Op, Op, Op,  Op,  —,  —,  —,  —,  —
   **Obs**,   —,  —,  Op, Op, Obs, Obs, —,  —,  —,  —,  —
   **Inv**,   —,  —,  Op, Op, Obs, Obs, —,  —,  —,  —,  —
   **S**,     —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **K**,     —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **J**,     —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **χ**,     —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **P**,     —,  —,  —,  —,  —,   —,  —,  —,  —,  —,  —

Scalar Multiplication (``a * b``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``*`` is *scalar* (complex number or JAX array) multiplication only — it is not defined
between two quantum objects.  The table below therefore has no quantum-object columns; it
shows the return type as a function of the left operand and the scalar type.

The symmetry ``scalar * obj`` (``__rmul__``) always delegates to ``obj * scalar``
(``__mul__``), so only one direction is shown.

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Type
     - ``obj * real_scalar``
     - ``obj * complex_scalar``
     - Notes
   * - **SV** / :class:`~quax.StateVector`
     - :class:`~quax.StateVector`
     - :class:`~quax.StateVector`
     -
   * - **DM** / :class:`~quax.DensityMatrix`
     - :class:`~quax.DensityMatrix`
     - :class:`~quax.DensityMatrix`
     -
   * - **U** / :class:`~quax.Unitary`
     - :class:`~quax.Operator`
     - :class:`~quax.Operator`
     - Scalar multiples of a unitary are not unitary
   * - **Op** / :class:`~quax.Operator`
     - :class:`~quax.Operator`
     - :class:`~quax.Operator`
     -
   * - **Obs** / :class:`~quax.Observable`
     - :class:`~quax.Observable`
     - :class:`~quax.Operator`
     - Complex multiples of a Hermitian operator are not Hermitian
   * - **Inv** / :class:`~quax.Involution`
     - :class:`~quax.Observable`
     - :class:`~quax.Operator`
     - Same promotion rules as Observable
   * - **S** / :class:`~quax.SuperOp`
     - :class:`~quax.SuperOp`
     - :class:`~quax.SuperOp`
     -
   * - **K** / :class:`~quax.KrausMap`
     - :class:`~quax.KrausMap`
     - :class:`~quax.KrausMap`
     -
   * - **J** / :class:`~quax.Choi`
     - :class:`~quax.Choi`
     - :class:`~quax.Choi`
     -
   * - **χ** / :class:`~quax.Chi`
     - :class:`~quax.Chi`
     - :class:`~quax.Chi`
     -
   * - **P** / :class:`~quax.PauliLiouville`
     - :class:`~quax.PauliLiouville`
     - :class:`~quax.PauliLiouville`
     -

Composition / Application (``a @ b``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``@`` is overloaded to mean operator composition, channel application, or inner product
depending on the operand types.

Special cases:

- **SV @ SV** returns a JAX :class:`~jax.Array` scalar (the inner product ⟨ψ|φ⟩), not a
  quantum object.
- When a superoperator acts on a :class:`~quax.StateVector`, the state is automatically
  promoted to a :class:`~quax.DensityMatrix` first.
- When composing two superoperators of *different* subtypes, the left operand is
  automatically converted to match the right operand's type.  For example, ``S @ K``
  converts the :class:`~quax.SuperOp` to a :class:`~quax.KrausMap` before composing,
  and returns a :class:`~quax.KrausMap`.
- :class:`~quax.Involution` shares its ``@`` implementation with :class:`~quax.Unitary`
  via Python's MRO, so it behaves identically to :class:`~quax.Unitary` in the table.
- :class:`~quax.Chi` operations raise :exc:`NotImplementedError` (**NI**) for all operand
  combinations.
- ``Array`` in the **SV @ SV** cell denotes a plain JAX scalar array.

.. csv-table::
   :header: "@",      SV,    DM,  U,  Op, Obs, Inv, S,  K,  J,  χ,  P
   :widths:  14, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
   :stub-columns: 1

   **SV**,    Array, SV,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **DM**,    SV,   DM,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **U**,     SV,   DM,  U,  Op, Op,  U,   S,  K,  J,  —,  P
   **Op**,    —,    —,   Op, Op, Op,  Op,  —,  —,  —,  —,  —
   **Obs**,   —,    —,   Op, Op, Op,  Op,  —,  —,  —,  —,  —
   **Inv**,   SV,   DM,  U,  Op, Op,  U,   S,  K,  J,  —,  P
   **S**,     DM,   DM,  S,  —,  —,   S,   S,  K,  J,  —,  P
   **K**,     DM,   DM,  K,  —,  —,   K,   S,  K,  J,  —,  P
   **J**,     DM,   DM,  J,  —,  —,   J,   S,  K,  J,  —,  P
   **χ**,     NI,   NI,  NI, NI, NI,  NI,  NI, NI, NI, NI, NI
   **P**,     DM,   DM,  P,  —,  —,   P,   S,  K,  J,  —,  P

Tensor Product (``a | b``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

``|`` is the tensor product (⊗) operator.  States can only be combined with other states,
and operators / superoperators can only be combined within their own "world"; mixing the
two is not defined.

Preservation rules:

- **Inv ⊗ Inv → Inv** (tensor product of involutions is an involution).
- **Inv ⊗ Obs → Obs** / **Obs ⊗ Inv → Obs** (Hermiticity is preserved).
- Any mix of :class:`~quax.Unitary` / :class:`~quax.Operator` with a non-Hermitian
  partner degrades to **Op**.
- When combining superoperators of *different* subtypes, the left operand is converted to
  the right operand's type (same rule as ``@``).
- :class:`~quax.Involution` defines a custom ``__or__`` that knows only about
  :class:`~quax.Operator`/Observable subtypes; combining an :class:`~quax.Involution`
  with a superoperator type on the right is therefore **—**.
- :class:`~quax.Chi` tensor products raise :exc:`NotImplementedError` (**NI**).

.. csv-table::
   :header: "|",      SV, DM, U,  Op, Obs, Inv, S,  K,  J,  χ,  P
   :widths:  14, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
   :stub-columns: 1

   **SV**,    SV,  DM,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **DM**,    DM,  DM,  —,  —,  —,   —,  —,  —,  —,  —,  —
   **U**,     —,   —,   U,  Op, Op,  U,   S,  K,  J,  —,  P
   **Op**,    —,   —,   Op, Op, Op,  Op,  —,  —,  —,  —,  —
   **Obs**,   —,   —,   Op, Op, Obs, Obs, —,  —,  —,  —,  —
   **Inv**,   —,   —,   Op, Op, Obs, Inv, —,  —,  —,  —,  —
   **S**,     —,   —,   S,  —,  —,   S,   S,  K,  J,  —,  P
   **K**,     —,   —,   K,  —,  —,   K,   S,  K,  J,  —,  P
   **J**,     —,   —,   J,  —,  —,   J,   S,  K,  J,  —,  P
   **χ**,     —,   —,   NI, NI, NI,  NI,  NI, NI, NI, NI, NI
   **P**,     —,   —,   P,  —,  —,   P,   S,  K,  J,  —,  P

Notes on Automatic Promotion
-----------------------------

Several of the table entries rely on silent *promotion* of the input type before the
operation is performed.  The most common promotions are:

- **StateVector → DensityMatrix**: when a :class:`~quax.SuperOperator` is applied to a
  :class:`~quax.StateVector` via ``@``.
- **Unitary → SuperOp / KrausMap / Choi / PauliLiouville**: when a
  :class:`~quax.Unitary` appears on the right of a superoperator in ``@`` or ``|``,
  the unitary is promoted to the matching superoperator representation.
- **Left superoperator → type of right superoperator**: when composing (``@``) or
  tensoring (``|``) superoperators of different subtypes, the left operand is converted to
  the right operand's type before the operation.
