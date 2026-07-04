Quickstart
==========

This guide introduces the core concepts and basic usage of Quax.

Basic Example
-------------

.. code-block:: python

   import jax
   import quax as qx
   
   # Create a quantum state
   state = qx.zero_state_vector(dims=(2,))
   
   # Apply a unitary operation
   U = qx.random_unitary(dims=((2,), (2,)), key=jax.random.PRNGKey(0))
   final_state = qx.apply_unitary_to_state_vector(U, state)

Creating Quantum States
-----------------------

State Vectors
~~~~~~~~~~~~~

.. code-block:: python

   import quax as qx
   import jax
   
   # Zero state |0⟩
   state = qx.zero_state_vector(dims=(2,))
   
   # Random state
   random_state = qx.random_state_vector(dims=(2,), key=jax.random.PRNGKey(0))

Density Matrices
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Zero state density matrix |0⟩⟨0|
   rho = qx.zero_state_matrix(dims=(2,))
   
   # Maximally mixed state
   mixed = qx.mixed_state_matrix(dims=(2,))
   
   # Random density matrix
   random_rho = qx.random_density_matrix(rank=2, dims=(2,), key=jax.random.PRNGKey(0))

Working with Gates
------------------

Quax stores all quantum objects in **tensor format**, where each qudit dimension is preserved
as a separate axis. This enables efficient tensor network operations. You can access the
traditional matrix representation via the ``.matrix`` property.

.. code-block:: python

   import jax.numpy as jnp
   import quax as qx
   
   # Define Pauli matrices (in matrix form)
   X_matrix = jnp.array([[0, 1], [1, 0]], dtype=complex)
   Y_matrix = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
   Z_matrix = jnp.array([[1, 0], [0, -1]], dtype=complex)
   
   # Create unitaries from matrices using from_matrix
   U_x = qx.Unitary.from_matrix(X_matrix, dims=((2,), (2,)))
   U_y = qx.Unitary.from_matrix(Y_matrix, dims=((2,), (2,)))
   
   # Or create directly in tensor form (shape matches dims)
   # For a single qubit: data shape is (d_out, d_in) = (2, 2)
   U_z = qx.Unitary(data=Z_matrix, num_qubits=1)  # Already tensor-shaped
   
   # Access the matrix representation
   print(U_x.matrix)  # Shape: (2, 2)

Superoperator Conversions
--------------------------

.. code-block:: python

   import quax as qx
   import jax
   
   # Start with a unitary
   U = qx.random_unitary(dims=((2,), (2,)), key=jax.random.PRNGKey(0))
   
   # Convert to different representations
   choi = qx.unitary_to_choi(U)
   pauli_liouville = qx.to_pauli_liouville(choi)
   superop = qx.to_superop(pauli_liouville)
   kraus = qx.to_kraus(superop)

Composing Operations
--------------------

.. code-block:: python

   import quax as qx
   import jax
   
   # Compose unitaries
   U1 = qx.random_unitary(dims=((2,), (2,)), key=jax.random.PRNGKey(0))
   U2 = qx.random_unitary(dims=((2,), (2,)), key=jax.random.PRNGKey(1))
   U_composed = qx.compose_unitary(U1, U2)
   
   # Compose Kraus maps
   K1 = qx.to_kraus(qx.random_choi(dims=((2,), (2,)), rank=2, key=jax.random.PRNGKey(0)))
   K2 = qx.to_kraus(qx.random_choi(dims=((2,), (2,)), rank=2, key=jax.random.PRNGKey(1)))
   K_composed = qx.compose_kraus_map(K1, K2)

Fidelity Calculations
---------------------

.. code-block:: python

   import quax as qx
   import jax
   
   # State fidelity
   state1 = qx.random_state_vector(dims=(2,), key=jax.random.PRNGKey(0))
   state2 = qx.random_state_vector(dims=(2,), key=jax.random.PRNGKey(1))
   fid = qx.fidelity(state1, state2)
   
   # Process fidelity
   choi1 = qx.random_choi(dims=((2,), (2,)), rank=2, key=jax.random.PRNGKey(0))
   choi2 = qx.random_choi(dims=((2,), (2,)), rank=2, key=jax.random.PRNGKey(1))
   proc_fid = qx.process_fidelity(choi1, choi2)

Quantum Channels
----------------

.. code-block:: python

   import jax.numpy as jnp
   import quax as qx
   
   # Depolarizing channel
   p = 0.1
   depol = qx.depolarizing_channel_superoperator(p, dims=(2,))
   
   # Thermal relaxation (T1 + pure dephasing) as a Lindbladian, evolved over the gate
   t1, tphi = 50e-6, 30e-6
   gate_time = 20e-9
   thermal = qx.evolve(qx.lindbladians.thermal_relaxation(t1, tphi), gate_time)

Multi-Qubit Systems
-------------------

.. code-block:: python

   import quax as qx
   import jax
   
   # Two-qubit state
   two_qubit_state = qx.zero_state_vector(dims=(2, 2))
   
   # Tensor product of operators
   U1 = qx.random_unitary(dims=((2,), (2,)), key=jax.random.PRNGKey(0))
   U2 = qx.random_unitary(dims=((2,), (2,)), key=jax.random.PRNGKey(1))
   U_tensor = qx.tensor_unitary(U1, U2)

Next Steps
----------

- Explore the :doc:`api/index` for detailed documentation
- Check the examples in the repository
- Learn about JAX's automatic differentiation with Quax
