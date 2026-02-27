Supported Operations
====================

This page provides detailed reference information about Quax's data representation
and supported operations on quantum objects.

Data Representation
-------------------

Quax stores all quantum objects in **tensor format**, preserving the structure of
individual qudits for efficient tensor network operations. The ``.matrix`` property
provides the flattened matrix representation when needed.

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

.. list-table:: Quantum Object Shapes
   :header-rows: 1
   :widths: 25 40 35

   * - Type
     - Tensor Shape
     - Matrix Shape
   * - StateVector
     - ``(*ensemble, d0, d1, ...)``
     - ``(*ensemble, prod(dims))``
   * - DensityMatrix
     - ``(*ensemble, d0_out, ..., d0_in, ...)``
     - ``(*ensemble, prod(dims), prod(dims))``
   * - Unitary/Operator/Observable
     - ``(*ensemble, d0_out, ..., d0_in, ...)``
     - ``(*ensemble, prod(dims_out), prod(dims_in))``
   * - KrausMap
     - ``(*ensemble, num_kraus, d0_out, ..., d0_in, ...)``
     - ``(*ensemble, num_kraus, d_out, d_in)``
   * - SuperOp/Choi/PauliLiouville
     - ``(*ensemble, d_out_bra..., d_out_ket..., d_in_bra..., d_in_ket...)``
     - ``(*ensemble, prod(dims_out)², prod(dims_in)²)``

Supported Operations on Quantum Objects
---------------------------------------

Unary Operations
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 8 11 8 8 8 10 8 6 13

   * - Operation
     - StateVector
     - DensityMatrix
     - Unitary
     - Operator
     - SuperOp
     - KrausMap
     - Choi
     - Chi
     - PauliLiouville
   * - ``-x`` (negation)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``x * scalar``
     - ✓
     - ✓
     - ✓¹
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``x.conj()``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``x.T`` (transpose)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``x.h`` (hermitian)
     - ✓³
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``x ** n`` (power)
     - ✗
     - ✓
     - ✓
     - ✗
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓

| ¹ Returns ``Unitary`` if ``|scalar| = 1``, otherwise ``Operator``
| ³ Returns ``conj()`` for vectors

Binary Operations: Composition (``@``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@`` operator composes quantum operations. The following table shows all
supported combinations and their output types.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Operation
     - Supported
     - Output Type
   * - ``StateVector @ StateVector``
     - ✓
     - scalar
   * - ``StateVector @ DensityMatrix``
     - ✓
     - StateVector
   * - ``StateVector @ Unitary``
     - ✗
     - —
   * - ``StateVector @ SuperOp``
     - ✗
     - —
   * - ``StateVector @ KrausMap``
     - ✗
     - —
   * - ``StateVector @ Choi``
     - ✗
     - —
   * - ``StateVector @ PauliLiouville``
     - ✗
     - —
   * - ``DensityMatrix @ StateVector``
     - ✓
     - StateVector
   * - ``DensityMatrix @ DensityMatrix``
     - ✓
     - DensityMatrix
   * - ``DensityMatrix @ Unitary``
     - ✗
     - —
   * - ``DensityMatrix @ SuperOp``
     - ✗
     - —
   * - ``DensityMatrix @ KrausMap``
     - ✗
     - —
   * - ``DensityMatrix @ Choi``
     - ✗
     - —
   * - ``DensityMatrix @ PauliLiouville``
     - ✗
     - —
   * - ``Unitary @ StateVector``
     - ✓
     - StateVector
   * - ``Unitary @ DensityMatrix``
     - ✓
     - DensityMatrix
   * - ``Unitary @ Unitary``
     - ✓
     - Unitary
   * - ``Unitary @ SuperOp``
     - ✓
     - SuperOp
   * - ``Unitary @ KrausMap``
     - ✓
     - KrausMap
   * - ``Unitary @ Choi``
     - ✓
     - Choi
   * - ``Unitary @ PauliLiouville``
     - ✓
     - PauliLiouville
   * - ``SuperOp @ StateVector``
     - ✓
     - DensityMatrix
   * - ``SuperOp @ DensityMatrix``
     - ✓
     - DensityMatrix
   * - ``SuperOp @ Unitary``
     - ✓
     - SuperOp
   * - ``SuperOp @ SuperOp``
     - ✓
     - SuperOp
   * - ``SuperOp @ KrausMap``
     - ✓
     - KrausMap
   * - ``SuperOp @ Choi``
     - ✓
     - Choi
   * - ``SuperOp @ PauliLiouville``
     - ✓
     - PauliLiouville
   * - ``KrausMap @ StateVector``
     - ✓
     - DensityMatrix
   * - ``KrausMap @ DensityMatrix``
     - ✓
     - DensityMatrix
   * - ``KrausMap @ Unitary``
     - ✓
     - KrausMap
   * - ``KrausMap @ SuperOp``
     - ✓
     - SuperOp
   * - ``KrausMap @ KrausMap``
     - ✓
     - KrausMap
   * - ``KrausMap @ Choi``
     - ✓
     - Choi
   * - ``KrausMap @ PauliLiouville``
     - ✓
     - PauliLiouville
   * - ``Choi @ StateVector``
     - ✓
     - DensityMatrix
   * - ``Choi @ DensityMatrix``
     - ✓
     - DensityMatrix
   * - ``Choi @ Unitary``
     - ✓
     - Choi
   * - ``Choi @ SuperOp``
     - ✓
     - SuperOp
   * - ``Choi @ KrausMap``
     - ✓
     - KrausMap
   * - ``Choi @ Choi``
     - ✓
     - Choi
   * - ``Choi @ PauliLiouville``
     - ✓
     - PauliLiouville
   * - ``PauliLiouville @ StateVector``
     - ✓
     - DensityMatrix
   * - ``PauliLiouville @ DensityMatrix``
     - ✓
     - DensityMatrix
   * - ``PauliLiouville @ Unitary``
     - ✓
     - PauliLiouville
   * - ``PauliLiouville @ SuperOp``
     - ✓
     - SuperOp
   * - ``PauliLiouville @ KrausMap``
     - ✓
     - KrausMap
   * - ``PauliLiouville @ Choi``
     - ✓
     - Choi
   * - ``PauliLiouville @ PauliLiouville``
     - ✓
     - PauliLiouville

Binary Operations: Tensor Product (``|``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``|`` operator computes tensor products. The following table shows all
supported combinations and their output types.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Operation
     - Supported
     - Output Type
   * - ``StateVector | StateVector``
     - ✓
     - StateVector
   * - ``StateVector | DensityMatrix``
     - ✓
     - DensityMatrix
   * - ``StateVector | Unitary``
     - ✗
     - —
   * - ``StateVector | SuperOp``
     - ✗
     - —
   * - ``StateVector | KrausMap``
     - ✗
     - —
   * - ``StateVector | Choi``
     - ✗
     - —
   * - ``StateVector | PauliLiouville``
     - ✗
     - —
   * - ``DensityMatrix | StateVector``
     - ✓
     - DensityMatrix
   * - ``DensityMatrix | DensityMatrix``
     - ✓
     - DensityMatrix
   * - ``DensityMatrix | Unitary``
     - ✗
     - —
   * - ``DensityMatrix | SuperOp``
     - ✗
     - —
   * - ``DensityMatrix | KrausMap``
     - ✗
     - —
   * - ``DensityMatrix | Choi``
     - ✗
     - —
   * - ``DensityMatrix | PauliLiouville``
     - ✗
     - —
   * - ``Unitary | StateVector``
     - ✗
     - —
   * - ``Unitary | DensityMatrix``
     - ✗
     - —
   * - ``Unitary | Unitary``
     - ✓
     - Unitary
   * - ``Unitary | SuperOp``
     - ✓
     - SuperOp
   * - ``Unitary | KrausMap``
     - ✓
     - KrausMap
   * - ``Unitary | Choi``
     - ✓
     - Choi
   * - ``Unitary | PauliLiouville``
     - ✓
     - PauliLiouville
   * - ``SuperOp | StateVector``
     - ✗
     - —
   * - ``SuperOp | DensityMatrix``
     - ✗
     - —
   * - ``SuperOp | Unitary``
     - ✓
     - SuperOp
   * - ``SuperOp | SuperOp``
     - ✓
     - SuperOp
   * - ``SuperOp | KrausMap``
     - ✓
     - KrausMap
   * - ``SuperOp | Choi``
     - ✓
     - Choi
   * - ``SuperOp | PauliLiouville``
     - ✓
     - PauliLiouville
   * - ``KrausMap | StateVector``
     - ✗
     - —
   * - ``KrausMap | DensityMatrix``
     - ✗
     - —
   * - ``KrausMap | Unitary``
     - ✓
     - KrausMap
   * - ``KrausMap | SuperOp``
     - ✓
     - SuperOp
   * - ``KrausMap | KrausMap``
     - ✓
     - KrausMap
   * - ``KrausMap | Choi``
     - ✓
     - Choi
   * - ``KrausMap | PauliLiouville``
     - ✓
     - PauliLiouville
   * - ``Choi | StateVector``
     - ✗
     - —
   * - ``Choi | DensityMatrix``
     - ✗
     - —
   * - ``Choi | Unitary``
     - ✓
     - Choi
   * - ``Choi | SuperOp``
     - ✓
     - SuperOp
   * - ``Choi | KrausMap``
     - ✓
     - KrausMap
   * - ``Choi | Choi``
     - ✓
     - Choi
   * - ``Choi | PauliLiouville``
     - ✓
     - PauliLiouville
   * - ``PauliLiouville | StateVector``
     - ✗
     - —
   * - ``PauliLiouville | DensityMatrix``
     - ✗
     - —
   * - ``PauliLiouville | Unitary``
     - ✓
     - PauliLiouville
   * - ``PauliLiouville | SuperOp``
     - ✓
     - SuperOp
   * - ``PauliLiouville | KrausMap``
     - ✓
     - KrausMap
   * - ``PauliLiouville | Choi``
     - ✓
     - Choi
   * - ``PauliLiouville | PauliLiouville``
     - ✓
     - PauliLiouville

Notes
~~~~~

- **Chi** is not included in binary operations because it has no implemented
  transformations to/from other representations
- The composition rules follow the principle that when mixing representations,
  the result uses the "right" operand's representation type
- State-Operator tensor products (``State | Operator``) return ``NotImplemented``
- Operator-State tensor products (``Operator | State``) return ``NotImplemented``
