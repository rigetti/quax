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