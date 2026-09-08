# Agent Instructions for Quax

This document provides guidance for AI agents working with the Quax codebase.

## Project Overview

**Quax** is a high-performance quantum information science library built on top of JAX. It provides tools for:
- Quantum state manipulation (state vectors and density matrices)
- Quantum gates and unitary operations
- Superoperator representations (Kraus, Choi, Pauli-Liouville, SuperOp)
- Quantum channels (depolarizing, thermal relaxation, bit flip, etc.)
- Distance metrics and fidelity measures
- Quantum operator transformations and compositions

## Technology Stack

- **Python**: 3.12+
- **Core Framework**: JAX (for automatic differentiation and GPU acceleration)
- **Testing**: pytest
- **Linting**: ruff, pyright
- **Documentation**: Sphinx with Furo theme
- **Build System**: poetry-core

## Code Organization

```
src/quax/                              # Main package
├── __init__.py                        # Public API exports
├── gates.py                           # Common quantum gates and measurement instruments (public submodule)
├── states.py                          # Predefined states (public submodule)
├── ensembles.py                       # Predefined ensembles (public submodule)
├── _apply.py                          # Applying operators to states
├── _apply_superoperator.py            # Superoperator application logic
├── channels.py                        # SuperOp channels + measurement instrument constructors (public submodule)
├── _circuits.py                       # Circuits, parameterized gates, merge planning
├── _compose.py                        # Operator composition
├── _errors.py                         # Structured circuit errors (kind-discriminated)
├── _metrics.py                        # Fidelity and distance functions
├── _mul.py                            # Scalar multiplication logic
├── _observables.py                    # Observable utilities
├── _operator_basis.py                 # Operator basis construction
├── _power.py                          # Operator exponent operations
├── _promotion.py                      # State promotion utilities
├── _quantum_objects.py                # Core quantum types (State, Operator, QuantumInstrument, etc.)
├── _simulators.py                     # State-vector and density-matrix simulators
├── _random.py                         # Random quantum objects
├── _state.py                          # State creation and manipulation
├── _superoperator_transformations.py  # Convert between representations
├── _tensor.py                         # Tensor product operations
├── _validation.py                     # Validation utilities
└── _visualization.py                  # Plotting functions

tests/                                 # Test suite
├── test_*.py                          # Unit tests for each module
├── instrument_helpers.py              # Shared helpers for instrument tests
├── reference_pauli_liouville.py       # Reference implementations
└── conftest.py                        # Pytest configuration

docs/                                  # Sphinx documentation
└── api/                               # API reference
```

Most public functions are exported from the top level of the package. However, gates, states, and ensembles are organized into public submodules:
- **gates**: Access quantum gates via `qx.gates.X`, `qx.gates.CNOT`, `qx.gates.MEASURE()`, etc.
- **states**: Access predefined states via `qx.states.KET0`, `qx.states.XPLUS`, etc.

The private submodules (those starting with an underscore) are organized for development convenience rather than providing a subpackage architecture.
## Development Workflow

### Setting Up the Environment

```bash
# From the project root...

# Install dependencies
poetry install
```

### Makefile recipes

The makefile contains recipes for common development tasks. For example

#### Run tests

Note that tests should always be run with jax at 64bit precision. This can be done by settings the environment variable

```bash
JAX_ENABLE_X64=1
```

Run the tests with 

```bash
make test-package
```

Tests are run with pytest and each file typically has a corresponding test file.

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_apply.py

# Run with verbose output
pytest -v
```

#### Test example notebooks

```bash
make test-examples
```

#### Format code

```bash
make format
```

#### Check types

```bash
make check-types
```

#### Check format

```bash
make check-format
```

#### Building Documentation

```bash
cd docs
make html
```

## Coding Conventions

### Style Guidelines

1. **Line Length**: 120 characters max
2. **Type Hints**: Required for all public functions
3. **Imports**: JAX imports should use `import jax` and `import jax.numpy as jnp`. In-package imports should be relative ex. `from ._quantum_objects import Unitary`
In tests or example notebooks, always use `import quax as qx` and then acccess functions using `qx.function`.
4. **Private Modules**: Internal implementation files use `_` prefix
5. **Docstrings**: Required for all public functions, use ReST format

### Naming Conventions

- **Functions**: `snake_case` (e.g., `apply_unitary_to_state_vector`)
- **Classes**: `PascalCase` (e.g., `StateVector`, `DensityMatrix`)
- **Constants**: `UPPER_CASE` (e.g., `KRAUS_OPS`)
- **Private functions**: `_snake_case` prefix

### JAX Best Practices

1. **Pure Functions**: All functions should be JAX-compatible (pure, no side effects)
2. **Array Operations**: Use `jax.numpy` instead of `numpy`
3. **JIT Compilation**: Functions should be compatible with `jax.jit`
4. **PRNG Keys**: Use `jax.random.PRNGKey` for randomness. Use the more modern `jax.random.key(seed)` for initializing keys.
5. **Shape Handling**: Use `dims` tuples to specify system dimensions

### Quantum Computing Conventions

A central task of the package is tracking the qudit dimensions. States have a single dimension for each qudit, while operators have both input and output dimensions which may sometimes differ (for example if a qutrit is promoted to a quart). The dimensions are represented as tuple of integers, for example (2, 2) for a 2-qubit state or ((2, 2), (2, 2)) for 2-qubit operator. Operator dimensions are (d_out, d_in).

Qubit: 2-dimensional qubit
Qutrit: 3-dimensional qubit
Quart: 4-dimensional qubit
Qudit: d-dimensional qubit

#### Data Representation: Tensor Format

All quantum objects store their data in **tensor format**, preserving the structure of individual qudits. This enables efficient tensor network operations and makes qudit indexing natural. The `.matrix` property provides the flattened matrix representation when needed.

1. **State Vectors**: Tensor with shape `(*ensemble, d0, d1, ...)` where each `di` is a qudit dimension
   - Example: 2-qubit state has shape `(2, 2)`
   - `.matrix` returns shape `(*ensemble, prod(dims))`

2. **Density Matrices**: Tensor with shape `(*ensemble, d0_out, d1_out, ..., d0_in, d1_in, ...)`
   - Example: 2-qubit density matrix has shape `(2, 2, 2, 2)`
   - `.matrix` returns shape `(*ensemble, prod(dims), prod(dims))`

3. **Unitaries/Operators**: Tensor with shape `(*ensemble, d0_out, d1_out, ..., d0_in, d1_in, ...)`
   - Example: 2-qubit unitary has shape `(2, 2, 2, 2)`
   - `.matrix` returns shape `(*ensemble, prod(dims_out), prod(dims_in))`

4. **Kraus Maps**: Tensor with shape `(*ensemble, num_kraus, d0_out, d1_out, ..., d0_in, d1_in, ...)`
   - Example: single-qubit Kraus map with 4 operators has shape `(4, 2, 2)`
   - `.matrix` returns shape `(*ensemble, num_kraus, d_out, d_in)`

5. **Superoperators** (SuperOp, Choi, PauliLiouville): Tensor with 4 groups of dimensions:
   `(*ensemble, d0_out_bra, ..., d0_out_ket, ..., d0_in_bra, ..., d0_in_ket, ...)`
   - Example: single-qubit superoperator has shape `(2, 2, 2, 2)`
   - `.matrix` returns shape `(*ensemble, prod(dims_out)**2, prod(dims_in)**2)`

6. **QuantumInstruments**: Tensor with an outcome axis followed by 4 groups of dimensions:
   `(*ensemble, num_outcomes, d0_out_bra, ..., d0_out_ket, ..., d0_in_bra, ..., d0_in_ket, ...)`
   - Each outcome slice is a CP (but not TP) superoperator; the sum over outcomes is CPTP.
   - Example: single-qubit ideal measurement has shape `(2, 2, 2, 2, 2)`
   - `.matrix` returns shape `(*ensemble, num_outcomes, prod(dims_out)**2, prod(dims_in)**2)`
   - Key properties: `confusion_matrix`, `transition_matrix`, `measured_qudits`
   - Supports `@` (compose) and `|` (tensor product) operators.

## Common Tasks

### Adding a New Quantum Gate

1. Add implementation to `src/quax/gates.py`
2. Gate will be automatically available via `qx.gates.GATE_NAME`
3. Add tests to `tests/test_quantum_objects.py`
4. Document in appropriate `.rst` file

### Adding a New Channel

Common *noise* channels are defined canonically as Lindbladian generators in
`src/quax/lindbladians.py` (rate-parameterized; obtain the CPTP channel via `qx.evolve(L, t)`).
Prefer adding a new noise channel there.

1. For a noise generator: add a factory to `src/quax/lindbladians.py` returning a `Lindbladian`.
2. For a SuperOp channel or a fixed non-divisible channel / instrument (no Lindbladian generator):
   implement in `src/quax/channels.py`.
3. Export from `__init__.py`.
4. Add tests to `tests/test_lindbladian.py`, `tests/test_channels.py`, or
   `tests/test_common_channels.py` as appropriate.

### Circuits and simulation

The circuit layer is deliberately free of any source language: a `ConstantCircuit` has no gate names,
no classical memory and no control flow, and building one from Quil (or anything else) is the
caller's job. Three types carry the layer:

- **`ConstantCircuit`** — concrete operators placed on a register. This is what every other quax
  operation consumes (`to_superops`, `compose`, `MergePlan.apply`).
- **`Circuit`** — the same structure, but a gate may be a `ParameterizedGate` that is
  built later from a flat parameter vector. `bind(params)` is the only bridge to `ConstantCircuit`.
- **`MergePlan`** — decides from the subsystems alone which operations may be fused. A plan is
  data, not a closure, which is what lets a simulator read the group layout (`flat_order`,
  `group_start`, `bases`) and build a fused operator stack without materialising operators.

Naming follows one rule: **the general union gets the plain name, the specific case gets the
qualifier.** So `CircuitOp` is anything an operation may be (built or not) and `ConstantOp` is
the already-built case; `Placement` is the general pair and `ConstantPlacement` the built one.
A `ParameterizedGate` argument is a `Slot` or a `Constant`, never a bare `int | float` — `bool`
is an `int`, an integer-valued constant is indistinguishable from a slot index, and `NewType` is
erased at runtime, so the bare version silently builds a different circuit.

Two invariants are load-bearing and are enforced rather than assumed:

1. **One parameter slot per `ParameterizedGate` argument.** No sharing, ever. This makes
   `jax.grad(...)[k]` refer to one gate occurrence instead of an implicit sum over several. A
   front end that shares a source parameter across gates maps it onto several slots and lets
   the chain rule sum them back. Nothing downstream would catch a violation — a shared slot
   builds fine and the forward pass stays correct — so `Circuit.__post_init__` checks
   that the slots are a permutation of `range(num_params)`.
2. **Simulator preparation is eager.** `Simulator.prepare()` runs at construction. Deferring it
   would let the first `compute` build cached values *inside* a JAX trace; the next call would
   then fail with a leaked-tracer error far from the cause.

Performance in the simulators is about the size of the traced graph, not the speed of the
arithmetic. Gate calls sharing a `batch_key` and an embedding shape are built under one
`jax.vmap`, and merged operations dispatch through a `jax.lax.switch` keyed by subsystem, so
the graph scales with the number of distinct gate *kinds* and *subsystems* rather than with the
number of operations. Building the stack the obvious way instead — a comprehension over the
bound circuit — puts one traced subgraph per gate into the jaxpr and XLA compile time then
grows superlinearly with depth. There is a regression test for this
(`test_the_traced_graph_does_not_grow_with_circuit_depth`); if it starts failing, batching has
been broken, even though every numerical test will still pass.

### Adding a New Transformation

1. Implement conversion in `src/quax/_superoperator_transformations.py`
2. Follow naming pattern: `{source}_to_{target}` (e.g., `choi_to_kraus`)
3. Add tests to `tests/test_superoperator_transformations.py`
4. Ensure inverse transformation exists when applicable

### Adding a New Distance Metric

1. Implement in `src/quax/_metrics.py`
2. Export from `__init__.py`
3. Add tests to `tests/test_metrics.py`
4. Include validation for input types

### Adding citations

Citing papers in the docstrings or comments is a good practice. We use the following format.
The title is abbreviated by a handful of letters and am arxiv URL is included wherever possible.

```
.. [GRAPTN] Tensor networks and graphical calculus for open quantum systems.
         Wood et al.
         Quant. Inf. Comp. 15, 0579-0811 (2015).
         (no DOI)
         https://arxiv.org/abs/1111.6950
```

There is currently no centralized bibliography, but that may change in the future.

## Testing Guidelines

### Test Structure

- Use pytest fixtures defined in `conftest.py`
- Test files mirror source file structure (`test_apply.py` ↔ `_apply.py`)
- Use `reference_pauli_liouville.py` for reference implementations

### Test Categories

1. **Unit Tests**: Test individual functions in isolation
2. **Integration Tests**: Test composed operations
3. **Validation Tests**: Verify mathematical properties (unitarity, CPTP, etc.)
4. **Comparison Tests**: Compare against reference implementations (QuTiP)

### Writing Tests

```python
import jax
import jax.numpy as jnp
import quax as qx

def test_example():
    # Setup
    key = jax.random.PRNGKey(42)
    state = qx.zero_state_vector(dims=(2,))
    
    # Action
    U = qx.random_unitary(dims=(2,), key=key)
    result = qx.apply_unitary_to_state_vector(U, state)
    
    # Assert
    assert jnp.allclose(jnp.linalg.norm(result), 1.0)
    
    # Using gates from the gates submodule
    state_after_hadamard = qx.apply_unitary_to_state_vector(qx.gates.H, state)
```

## Common Pitfalls

1. **JAX Arrays are Immutable**: Don't use in-place operations
2. **Random Number Generation**: Always pass `key` parameter explicitly
3. **Shape Dimensions**: Use `dims` tuples, not direct array shapes
4. **Type Annotations**: Use `jax.Array` not `np.ndarray`
5. **Imports**: Don't import from private modules (`_*.py`) outside the package

## Dependencies and References

### Core Dependencies
- **JAX**: Automatic differentiation and GPU acceleration
- **QuTiP**: Used for testing/validation (dev dependency only)

### Mathematical References
- Nielsen, M. A., & Chuang, I. L. (2001). Quantum computation and quantum information (Vol. 2). Cambridge: Cambridge university press.
- Hashim, A., Nguyen, L. B., Goss, N., Marinelli, B., Naik, R. K., Chistolini, T., ... & Siddiqi, I. (2024). A practical introduction to benchmarking and characterization of quantum computers. arXiv preprint arXiv:2408.12064.
- Wood, C. J., Biamonte, J. D., & Cory, D. G. (2011). Tensor networks and graphical calculus for open quantum systems. arXiv preprint arXiv:1111.6950.

### Useful online references
- [Forest benchmarking documention](https://forest-benchmarking.readthedocs.io/en/latest/superoperator_representations.html)
- [Qutip documentation](https://qutip.readthedocs.io/en/qutip-5.2.x/guide/guide-states.html)
- [Jax documentation](https://docs.jax.dev/en/latest/index.html)
- [Trueq documentation](https://trueq.quantumbenchmark.com/api/math.html)

## Working with AI Agents

### When Modifying Code

1. **Always run tests** after making changes
2. **Check type hints** with pyright
3. **Verify JAX compatibility** (no NumPy-specific operations)
4. **Update docstrings** for public API changes
5. **Check exports** in `__init__.py` for new public functions
6. Use `complex` rather than `jnp.complex128` or `jnp.complex64`.

### When Adding Features

1. **Follow existing patterns** in similar functions
2. **Add comprehensive tests** including edge cases
3. **Document mathematical background** in docstrings
4. **Consider GPU compatibility** (avoid Python loops)
5. **Validate quantum properties** (unitarity, CPTP, trace preservation)

### When Debugging

1. **Check array shapes** with `jnp.shape`
2. **Verify numerical precision** with `jnp.allclose`
3. **Test with simple cases** (single qubit, known gates)
4. **Compare with QuTiP** when available
5. **Use `jax.debug.print`** for debugging JIT-compiled functions

## License

Copyright 2026 Rigetti & Co, LLC. Licensed under Apache License 2.0.

## Contact

For questions or issues, refer to the project maintainers at Rigetti Computing.
