# Quax

A high-performance quantum information science library built on top of JAX.

## Documentation

[Check out the docs.](https://rigetti.gitlab.io/application_benchmarking/quax)

## Features

- **Quantum Objects**: States, gates and superoperators objects are defined ot allow natural manipulations and operations.
- **Clear typing**: Standard objects such as Unitaries, Chois and Density matrices are all typed, clarifying the nature of various objects.
- **Standard operators** composition/application `@`, tensor products `|`, scalar multiplication `*` and powers `*` are defined on all quantum objects.
- **Automatic promotion** Pure states and operators are automatically promoted to mixed states and superoperators when appropriate.
- **Qudit support** Operations on d-dimensional qudits are supported.
- **Batch support** Operating on batches or ensembles of states is supported for straightforward parallelization.

## Installation

```bash
pip install rigetti-quax
```

## Quick Example

```python
import jax
import quax as qx

# Create a quantum state
state = qx.zero_state_vector(dims=(2,))

# Apply a unitary operation
U = qx.random_unitary(dims=(2,), key=jax.random.key(0))
final_state = U @ state

# Convert between representations
choi = qx.unitary_to_choi(U)
pauli_liouville = qx.to_pauli_liouville(choi)
```

## License

Copyright 2026 Rigetti & Co, LLC. Licensed under Apache License 2.0.
