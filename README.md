# Quax

A high-performance quantum information science library built on top of JAX.

## Data Representation

Quax stores all quantum objects in **tensor format**, preserving the structure of individual qudits for efficient tensor network operations. The `.matrix` property provides the flattened matrix representation when needed.

| Type | Tensor Shape | Matrix Shape |
|------|--------------|--------------|
| StateVector | `(*ensemble, d0, d1, ...)` | `(*ensemble, prod(dims))` |
| DensityMatrix | `(*ensemble, d0_out, ..., d0_in, ...)` | `(*ensemble, prod(dims), prod(dims))` |
| Unitary/Operator | `(*ensemble, d0_out, ..., d0_in, ...)` | `(*ensemble, prod(dims_out), prod(dims_in))` |
| KrausMap | `(*ensemble, num_kraus, d0_out, ..., d0_in, ...)` | `(*ensemble, num_kraus, d_out, d_in)` |
| SuperOp/Choi/PauliLiouville | `(*ensemble, d_out_bra..., d_out_ket..., d_in_bra..., d_in_ket...)` | `(*ensemble, prod(dims_out)², prod(dims_in)²)` |

## Supported Operations on Quantum Objects

### Unary Operations

| Operation | StateVector | DensityMatrix | Unitary | Kraus | SuperOp | KrausMap | Choi | Chi | PauliLiouville |
|-----------|-------------|---------------|---------|-------|---------|----------|------|-----|----------------|
| `-x` (negation) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `x * scalar` | ✓ | ✓ | ✓¹ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `x.conj()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `x.T` (transpose) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `x.h` (hermitian) | ✓³ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `x ** n` (power) | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ |

¹ Returns `Unitary` if `|scalar| = 1`, otherwise `Kraus`  
² Returns self (no-op for vectors)  
³ Returns `conj()` for vectors

---


### Binary Operations: Composition (`@`)

| Operation | Supported | Output Type |
|-----------|-----------|-------------|
| `StateVector @ StateVector` | ✓ | scalar |
| `StateVector @ DensityMatrix` | ✓ | StateVector |
| `StateVector @ Unitary` | ✗ | — |
| `StateVector @ SuperOp` | ✗ | — |
| `StateVector @ KrausMap` | ✗ | — |
| `StateVector @ Choi` | ✗ | — |
| `StateVector @ PauliLiouville` | ✗ | — |
| `DensityMatrix @ StateVector` | ✓ | StateVector |
| `DensityMatrix @ DensityMatrix` | ✓ | DensityMatrix |
| `DensityMatrix @ Unitary` | ✗ | — |
| `DensityMatrix @ SuperOp` | ✗ | — |
| `DensityMatrix @ KrausMap` | ✗ | — |
| `DensityMatrix @ Choi` | ✗ | — |
| `DensityMatrix @ PauliLiouville` | ✗ | — |
| `Unitary @ StateVector` | ✓ | StateVector |
| `Unitary @ DensityMatrix` | ✓ | DensityMatrix |
| `Unitary @ Unitary` | ✓ | Unitary |
| `Unitary @ SuperOp` | ✓ | SuperOp |
| `Unitary @ KrausMap` | ✓ | KrausMap |
| `Unitary @ Choi` | ✓ | Choi |
| `Unitary @ PauliLiouville` | ✓ | PauliLiouville |
| `SuperOp @ StateVector` | ✓ | DensityMatrix |
| `SuperOp @ DensityMatrix` | ✓ | DensityMatrix |
| `SuperOp @ Unitary` | ✓ | SuperOp |
| `SuperOp @ SuperOp` | ✓ | SuperOp |
| `SuperOp @ KrausMap` | ✓ | KrausMap |
| `SuperOp @ Choi` | ✓ | Choi |
| `SuperOp @ PauliLiouville` | ✓ | PauliLiouville |
| `KrausMap @ StateVector` | ✓ | DensityMatrix |
| `KrausMap @ DensityMatrix` | ✓ | DensityMatrix |
| `KrausMap @ Unitary` | ✓ | KrausMap |
| `KrausMap @ SuperOp` | ✓ | SuperOp |
| `KrausMap @ KrausMap` | ✓ | KrausMap |
| `KrausMap @ Choi` | ✓ | Choi |
| `KrausMap @ PauliLiouville` | ✓ | PauliLiouville |
| `Choi @ StateVector` | ✓ | DensityMatrix |
| `Choi @ DensityMatrix` | ✓ | DensityMatrix |
| `Choi @ Unitary` | ✓ | Choi |
| `Choi @ SuperOp` | ✓ | SuperOp |
| `Choi @ KrausMap` | ✓ | KrausMap |
| `Choi @ Choi` | ✓ | Choi |
| `Choi @ PauliLiouville` | ✓ | PauliLiouville |
| `PauliLiouville @ StateVector` | ✓ | DensityMatrix |
| `PauliLiouville @ DensityMatrix` | ✓ | DensityMatrix |
| `PauliLiouville @ Unitary` | ✓ | PauliLiouville |
| `PauliLiouville @ SuperOp` | ✓ | SuperOp |
| `PauliLiouville @ KrausMap` | ✓ | KrausMap |
| `PauliLiouville @ Choi` | ✓ | Choi |
| `PauliLiouville @ PauliLiouville` | ✓ | PauliLiouville |

---

### Binary Operations: Tensor Product (`|`)

| Operation | Supported | Output Type |
|-----------|-----------|-------------|
| `StateVector \| StateVector` | ✓ | StateVector |
| `StateVector \| DensityMatrix` | ✓ | DensityMatrix |
| `StateVector \| Unitary` | ✗ | — |
| `StateVector \| SuperOp` | ✗ | — |
| `StateVector \| KrausMap` | ✗ | — |
| `StateVector \| Choi` | ✗ | — |
| `StateVector \| PauliLiouville` | ✗ | — |
| `DensityMatrix \| StateVector` | ✓ | DensityMatrix |
| `DensityMatrix \| DensityMatrix` | ✓ | DensityMatrix |
| `DensityMatrix \| Unitary` | ✗ | — |
| `DensityMatrix \| SuperOp` | ✗ | — |
| `DensityMatrix \| KrausMap` | ✗ | — |
| `DensityMatrix \| Choi` | ✗ | — |
| `DensityMatrix \| PauliLiouville` | ✗ | — |
| `Unitary \| StateVector` | ✗ | — |
| `Unitary \| DensityMatrix` | ✗ | — |
| `Unitary \| Unitary` | ✓ | Unitary |
| `Unitary \| SuperOp` | ✓ | SuperOp |
| `Unitary \| KrausMap` | ✓ | KrausMap |
| `Unitary \| Choi` | ✓ | Choi |
| `Unitary \| PauliLiouville` | ✓ | PauliLiouville |
| `SuperOp \| StateVector` | ✗ | — |
| `SuperOp \| DensityMatrix` | ✗ | — |
| `SuperOp \| Unitary` | ✓ | SuperOp |
| `SuperOp \| SuperOp` | ✓ | SuperOp |
| `SuperOp \| KrausMap` | ✓ | KrausMap |
| `SuperOp \| Choi` | ✓ | Choi |
| `SuperOp \| PauliLiouville` | ✓ | PauliLiouville |
| `KrausMap \| StateVector` | ✗ | — |
| `KrausMap \| DensityMatrix` | ✗ | — |
| `KrausMap \| Unitary` | ✓ | KrausMap |
| `KrausMap \| SuperOp` | ✓ | SuperOp |
| `KrausMap \| KrausMap` | ✓ | KrausMap |
| `KrausMap \| Choi` | ✓ | Choi |
| `KrausMap \| PauliLiouville` | ✓ | PauliLiouville |
| `Choi \| StateVector` | ✗ | — |
| `Choi \| DensityMatrix` | ✗ | — |
| `Choi \| Unitary` | ✓ | Choi |
| `Choi \| SuperOp` | ✓ | SuperOp |
| `Choi \| KrausMap` | ✓ | KrausMap |
| `Choi \| Choi` | ✓ | Choi |
| `Choi \| PauliLiouville` | ✓ | PauliLiouville |
| `PauliLiouville \| StateVector` | ✗ | — |
| `PauliLiouville \| DensityMatrix` | ✗ | — |
| `PauliLiouville \| Unitary` | ✓ | PauliLiouville |
| `PauliLiouville \| SuperOp` | ✓ | SuperOp |
| `PauliLiouville \| KrausMap` | ✓ | KrausMap |
| `PauliLiouville \| Choi` | ✓ | Choi |
| `PauliLiouville \| PauliLiouville` | ✓ | PauliLiouville |

---

### Notes

- **Chi** is not included in binary operations because it has no implemented transformations to/from other representations
- The composition rules follow the principle that when mixing representations, the result uses the "right" operand's representation type
- State-Operator tensor products (`State | Operator`) return `NotImplemented`
- Operator-State tensor products (`Operator | State`) return `NotImplemented`
