# Quax

A high-performance quantum information science library built on top of JAX.

## Documentation

[Check out the docs.](https://rigetti.gitlab.io/application_benchmarking/quax)

## Features

- **Quantum Objects**: States, gates and superoperators objects are defined to allow natural manipulations and operations.
- **Clear typing**: Standard objects such as Unitaries, Chois and Density matrices are all typed, clarifying the nature of various objects.
- **Standard operators** composition/application `@`, tensor products `|`, scalar multiplication `*` and powers `*` are defined on all quantum objects.
- **Automatic promotion** Pure states and operators are automatically promoted to mixed states and superoperators when appropriate.
- **Quantum instruments** Mid-circuit measurements are modeled with `QuantumInstrument`, supporting confusion/transition matrices, composition, and tensor products.
- **Qudit support** Operations on d-dimensional qudits are supported.
- **Batch support** Operating on batches or ensembles of states is supported for straightforward parallelization.

## Installation

```bash
pip install rigetti-quax
```

## Quick Example

```python
import jax
import jax.numpy as jnp
import quax as qx

# Create a 2-qubit state
psi_0: qx.StateVector = qx.zero_state_vector(dims=(2, 2,))

# Apply gates to prepare a pure Bell state
psi = (qx.gates.H | qx.gates.I) @ psi_0
psi = qx.gates.CNOT @ psi

# Generate an ensemble of Lindbladians with varying strenght
L: qx.Lindbladian = qx.lindbladians.amplitude_damping(jnp.linspace(0.0, 0.1, 10))

# Combine the Lindbladians with the CNOT unitary to create an ensemble of quantum channels
# The unitary is automatically promoted to a superoperator to support the addition of the Lindbladian
noisy_cnot: qx.SuperOp = qx.gates.CNOT + (L | L)

# Apply the noise operation to create a noisy Bell states
rho: qx.DensityMatrix = noisy_cnot @ (qx.gates.H | qx.gates.I) @ psi_0

# Compute the fidelity between the noisy Bell state and the ideal Bell state
fidelity: jax.Array = qx.fidelity(rho, psi)

# Now let's apply a qutrit gate to the ensemble of noisy Bell states. 
rho = (qx.gates.TRX12(jnp.pi) | qx.gates.I) @ rho

# We can estimate the 𝜆8 observable
obesrvables: jax.Array = qx.estimate(rho, observable=(qx.gates.GELLMANN8) | qx.gates.I)
```

## Acknowledgements

Quax draws inspiration, educational material, and some code from [forest-benchmarking](https://github.com/rigetti/forest-benchmarking), Rigetti's open-source library for quantum characterization, verification, and validation. We gratefully acknowledge the forest-benchmarking contributors for their foundational work on superoperator representations, quantum channel conventions, and distance metrics.

## License

Copyright 2026 Rigetti & Co, LLC. Licensed under Apache License 2.0.
