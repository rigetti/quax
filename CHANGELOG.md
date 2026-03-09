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