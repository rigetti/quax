API Reference
=============

Quantum Objects
---------------

.. currentmodule:: quax

.. autosummary::
   :toctree: generated/
   :nosignatures:

   State
   StateVector
   DensityMatrix
   Operator
   SuperOperator
   Unitary
   KrausMap
   Choi
   PauliLiouville
   SuperOp

State Operations
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   zero_state_vector
   zero_state_matrix
   mixed_state_matrix
   tensor_state_vectors
   tensor_density_matrices
   promote_state_vector_to_density_matrix

Promotion
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   promote
   promote_incoherent
   embed
   permute

Apply Operations
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apply_unitary_to_state_vector
   apply_unitary_to_density_matrix
   apply_kraus_to_state_vector
   apply_kraus_to_density_matrix
   apply_choi_to_density_matrix
   apply_pauli_liouville_to_density_matrix
   apply_superop_to_density_matrix
   partial_trace
   targeted_apply_unitary
   targeted_apply_unitary_to_density_matrix
   targeted_apply_kraus_map
   targeted_apply_superop

Composition
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compose_unitary
   compose_kraus_map
   compose_choi
   compose_pauli_liouville
   compose_superop

Tensor Products
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tensor_state_vector
   tensor_density_matrix
   tensor_unitary
   tensor_kraus
   tensor_choi
   tensor_pauli_liouville
   tensor_superop

Superoperator Transformations
------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   to_kraus
   to_choi
   to_pauli_liouville
   to_superop
   unitary_to_choi
   unitary_to_pauli_liouville
   unitary_to_superop
   kraus_to_choi
   kraus_to_pauli_liouville
   kraus_to_superop
   choi_to_kraus
   choi_to_pauli_liouville
   choi_to_superop
   pauli_liouville_to_kraus
   pauli_liouville_to_choi
   pauli_liouville_to_superop
   superop_to_kraus
   superop_to_choi
   superop_to_pauli_liouville
   unitary_to_hamiltonian

Power Operations
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   power_unitary
   power_kraus
   power_choi
   power_pauli_liouville
   power_superop

Distance Metrics
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fidelity
   process_fidelity
   unitary_entanglement_fidelity
   average_fidelity_to_process_fidelity
   process_fidelity_to_average_fidelity
   depolarizing_constant_to_average_fidelity
   depolarizing_constant_to_process_fidelity
   average_fidelity_to_depolarizing_constant
   process_fidelity_to_depolarizing_constant
   unitarity_to_stochastic_infidelity

Random Generators
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   random_state_vector
   random_density_matrix
   random_unitary
   random_choi
   ginibre_matrix_complex

Common Channels
---------------

Rate-parameterized noise channels live in the :mod:`quax.channels` submodule (e.g.
``qx.channels.depolarizing``), each an evolved :mod:`quax.lindbladians` generator.  Measurement
instruments:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   instrument_from_axis
   instrument_from_confusion_and_transition

Observables Computation
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_kraus_observables_from_states
   compute_choi_observables_from_states
   compute_pauli_liouville_observables_from_states
   compute_superop_observables_from_states
