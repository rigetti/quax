# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

<!-- Instructions for developers:
1. The latest version comes first.
2. The release date of each version is displayed.
3. Group changes under the following sections:
  * **Added** for new features.
  * **Changed** for changes in existing functionality.
  * **Deprecated** for soon-to-be removed features.
  * **Removed** for now removed features.
  * **Fixed** for any bug fixes.
  * **Security** in case of vulnerabilities. -->

## [0.6.4] - 2026-06-23

### Added

- `promote`, `embed` and `permute` for promoting, embeding and permuting quantum objects.

## [0.6.3] - 2026-05-15

### Added

- `targeted_apply_unitary_to_density_matrix` applies unitaries to density matrices.


## [0.6.2] - 2026-05-14

### Added

- `state_vector_reduced_density_matrix` computes the reduced density matrix for a subsytem of a state vector.

### Changed

- `targeted_apply_kraus_map_trajectory` uses reduced density matrices to compute Bron probabilities, improving performance.

## [0.6.1] - 2026-05-06

### Added

- Benchmarks for targeted_apply for `KrausMap`, `QuantumInstrument` and `Superoperator`.
- `truncate_kraus` reduces the size of Kraus maps by truncating small operators.
- `cuda` group now available for running on Nvidia GPU. Install with `poetry install --with cuda`.

### Fixed

- `targeted_apply_kraus_map_trajectory` now broadcasts ensembles of keys and states.

## [0.6.0] - 2026-05-06

### Added

- `QuantumInstrument` quantum object for modeling mid-circuit measurements, with per-outcome superoperators, `confusion_matrix` and `transition_matrix` properties.
- `MEASURE(dim)` and `RESET(dim)` gate constructors in `qx.gates` for ideal projective measurement and reset channels.
- `instrument_from_confusion_and_transition` and `instrument_from_axis` constructors for building noisy instruments from classical error models.
- `apply_instrument_to_density_matrix`, `apply_instrument_to_state_vector`, `select_outcome`, and targeted variants for applying instruments to quantum states.
- `compose_instrument` and `tensor_instrument` for sequential composition (`@`) and tensor product (`|`) of instruments.
- `classification_fidelity`, `non_demolition_fidelity`, and `instrument_fidelity` metrics for characterizing instrument quality.
- `validate` support for `QuantumInstrument` (checks per-outcome CP and total TP).
- `plot` support for `QuantumInstrument` (per-outcome superoperator heatmaps).
- `DensityMatrix.pretty_print()` for human-readable `|i⟩⟨j|` display.
- `promote_hilbert_space` auto-converts `Unitary` to `SuperOp` when paired with a channel type to avoid global-phase artifacts.
- Quantum instruments documentation page (`docs/quantum-instruments.rst`) with comprehensive theory and usage guide.
- `docs/citations.bib` bibliography for quantum instrument references.

### Changed

- `process_fidelity` now auto-promotes operands via `promote_hilbert_space` when dimensions differ, instead of raising an error.
- Removed `title` parameter from `plot` and `plot_pauli_transfer_matrix` functions.
- Updated phase colorscale and conventions in density matrix visualization.

### Fixed

- `Unitary.__matmul__(KrausMap)` composition order: `U @ K` now correctly composes U (applied second) with K (applied first).
- Superoperator promotion now uses coherent extension (Kraus-based) instead of the previous zero-pad + complement projector approach, preserving coherences between original and complement subspaces.
- Parametric gates (`RZ`, `PHASEDRX`, `U`, `CPHASE00`, `RZZ`, `CAN`) now return `Unitary` instances instead of `Operator`.

## [0.5.3] - 2026-04-23

### Added

- `unitary` and `stochastic_infidelity` functions.

## [0.5.2] - 2026-04-07

### Added

- `plot` function for `DensityMatrix`, `StateVector`, `Superoperator`, `Operator`.

### Changed

- Update example notebooks and documentation.

## [0.5.1] - 2026-04-06

### Fixed

- Re-added and improved the example notebook.

## [0.5.0] - 2026-04-02

### Added

- **Qudit support**: All core operations (state creation, channel application, superoperator transformations, composition, targeted apply, promotion, distance metrics) now support arbitrary qudit dimensions, not just qubits.
- Qutrit gates: `TX` (shift), `TY`, `TZ` (clock), `TH` (Hadamard/QFT), `TSHIFT`, `TSWAP`.
- Gell-Mann matrices: `GELLMANN1`–`GELLMANN8` and `GELLMANN_MATRICES` ensemble as `Observable` instances.
- Weyl operators: `W00`–`W22` and `WEYLS3` ensemble.
- `qudit_operator_basis` and `n_qudit_basis` for constructing Weyl-Heisenberg operator bases for arbitrary qudit dimensions.
- `bitstring_probability` and `probabilities` for computing measurement outcome probabilities from state vectors and density matrices.
- `promote` (singledispatch) for embedding quantum objects into larger Hilbert spaces, supporting `StateVector`, `DensityMatrix`, `Unitary`, `Operator`, `SuperOp`, `Choi`, `KrausMap`, and `PauliLiouville`.
- `broadcast_qudits` utility for computing the target dims when composing objects of different sizes.
- Leakage channel operators: `leakage_operators`, `stochastic_leakage_operators`, `seepage_operators`.
- `depolarizing_channel_superoperator` now accepts a `dims` keyword argument for arbitrary qudit dimensions.
- Leakage randomized benchmarking example notebook.

### Changed

- Distance metric functions (`depolarizing_constant_to_average_fidelity`, `average_fidelity_to_process_fidelity`, `process_fidelity_to_average_fidelity`, `process_fidelity_to_depolarizing_constant`, `average_fidelity_to_depolarizing_constant`, `unitarity_to_stochastic_infidelity`, `depolarizing_constant_to_process_fidelity`) now accept an optional `dim` parameter for non-qubit systems.
- Test fixtures consolidated from separate `num_qubits`/`qudit_dim` parameters to a unified `dims` tuple covering qubit, qutrit, ququint, and mixed-dimension systems.
- Test suite expanded with qutrit and mixed qubit-qutrit test cases throughout.

## [0.4.4] - 2026-03-11

## Added

- `targeted_apply_kraus_map_trajectory` for Monte Carlo trajectory simulation: applies a Kraus map to a state vector probabilistically, sampling one outcome per ensemble element. Supports ensemble broadcasting of operators, states, and keys.

## [0.4.3] - 2026-03-09

## Added

- `targeted_apply` to apply superoperators to larger states.

## Changed

- `unitary_to_kraus` renamed to `unitary_to_kraus_map` to be consistent with other changed functions in version 4.0.

## [0.4.2] - 2026-03-09

## Added 

- `validate` will check if the quantum object meets is mathematically valid.

## [0.4.1] - 2026-03-09

## Added

- Example notebook `hamiltonians.ipynb`
- Functions `exp` and `cis`.

## Changed

- Gates are now defined as exponentials of Paulis.
- Parametric gates can be created with arrays of parameters.

## [0.4.0] - 2026-02-26

### Added

- `QuantumObject` base class that holds the shared data representation (`data`, `num_qubits`) and common operations (negation, scalar multiplication, pytree support, conjugation, ensemble indexing) for `State`, `Operator`, and `SuperOperator`.
- `Observable(Operator)` — Hermitian operator type (`A = A†`) with type-preserving algebra: real scalar multiplication and addition/subtraction of observables return `Observable`; complex scalar multiplication or mixing with a plain `Operator` returns `Operator`.
- `Involution(Observable, Unitary)` — self-inverse Hermitian unitary (`A² = I`) with fine-grained scalar multiplication (±1 → `Involution`, unit complex → `Unitary`, real → `Observable`, general → `Operator`).
- `estimate()` function for computing expectation values `⟨ψ|A|ψ⟩` and `Tr[Aρ]` of an `Observable` given a `StateVector` or `DensityMatrix`.
- `compose_operator()` for composing two `Operator` instances (with ensemble broadcasting). `compose_unitary` now delegates to it internally.
- `tensor_operator()`, `tensor_observable()`, `tensor_involution()` for tensor products at each type level. `tensor_unitary` now delegates to `tensor_operator` internally.
- `random_operator()` and `random_observable()` for generating random operators and Hermitian operators from the Ginibre ensemble.
- `power_observable()` for fractional powers of Hermitian matrices via eigendecomposition.
- 10 gate constants promoted to `Involution`: `I`, `X`, `Y`, `Z`, `H`, `CZ`, `CNOT`, `CCNOT`, `SWAP`, `CSWAP`.

### Changed

- `State`, `Operator`, and `SuperOperator` now inherit from `QuantumObject` instead of being independent base classes. `State` and `SuperOperator` are siblings of `Operator` (not subclasses).
- `Kraus` type removed and replaced by `Operator` throughout. Projection operators `P0`/`P1` in `gates.py` are now `Operator` instances.
- `compose_kraus` renamed to `compose_kraus_map`.
- `random_choi_BCSZ` renamed to `random_choi`.
- Common channel functions (`bit_flip_operators`, `phase_flip_operators`, `depolarizing_operators`, `amplitude_damping_operators`, `relaxation_operators`) now return `KrausMap` instead of tuples of `Kraus`.

## [0.3.1] - 2026-02-23

### Added

- `Operator` and `State` ensembles are now indexable.

## [0.3.0] - 2026-02-15

### Changed

- `num_ensemble_dimensions` is removed and `num_qubits` added to all quantum objects. This enables simpler vmapping over objects.

## [0.2.3] - 2026-02-13

- Python bound lowered to >=3.11

## [0.2.2] - 2026-01-31

- Update docs and README

## [0.2.1] - 2026-01-31

### Added

- Push to pypi

## [0.2.0] - 2026-01-22

### Added

- README.md
- CHANGELOG.md

### Changed

- All quantum objects now natively store data in tensor format. The matrix view is available through `.matrix`.

## [0.1.0] - 2026-01-19

### Added

- Created package
- Quantum objects (`Choi`, `StateVector`, `DensityMatrix`, `Unitary`, `PauliLiouville`, `Superop`, `Chi`)
- Composition and tensor functions
- Fidelities metrics
- Common channels