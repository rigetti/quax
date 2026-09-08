# Copyright 2026 Rigetti & Co, LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable simulators for :class:`~quax.ParametricCircuit`.

Two simulators share one evolution machinery:

* :class:`StateVectorSimulator` evolves a pure state under a unitary-only circuit.
* :class:`DensityMatrixSimulator` evolves a density matrix under any circuit, lifting every
  operation to a superoperator.

Both are jit- and grad-friendly::

    >>> simulator = StateVectorSimulator(circuit=circuit)
    >>> psi = jax.jit(simulator.compute)(params)
    >>> grad = jax.grad(lambda p: cost(simulator.compute(p)))(params)

There is no notion of a program here, and none of measurement outcomes, classical memory or
control flow: a simulator takes a circuit and returns a state.  Turning a source language
into a circuit belongs to whatever owns that language.

Two ideas carry all the performance, and both are about the size of the traced graph rather
than the speed of the arithmetic.

**Operations are built in batches, not one at a time.**  Every :class:`~quax.GateCall`
sharing a :attr:`~quax.GateCall.batch_key` and an embedding shape is built under a single
:func:`jax.vmap`, so the graph grows with the number of distinct gate *kinds* rather than the
number of gates.  Building the stack the obvious way — a comprehension over the bound circuit
— instead puts one traced subgraph per gate into the jaxpr, and XLA compile time then grows
superlinearly in circuit depth.

**Operations are applied through a switch, not unrolled.**  The merged stack is applied by a
:func:`jax.lax.scan` whose body dispatches each operator to the :func:`jax.lax.switch` branch
for its subsystem, so the graph grows with the number of distinct *subsystems* rather than
the number of operations.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, final, override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ._apply import targeted_apply_superop, targeted_apply_unitary
from ._circuits import Circuit, CircuitOp, GateCall, MergePlan, ParametricCircuit, ParametricOp
from ._errors import CircuitErrorKind, CircuitTypeError
from ._promotion import embed
from ._quantum_objects import DensityMatrix, State, StateVector, SuperOp, Unitary
from ._state import zero_state_matrix, zero_state_vector
from ._superoperator_transformations import to_superop

# ══════════════════════════════════════════════════════════
# Vectorised operator-stack construction
# ══════════════════════════════════════════════════════════


def _embed_and_pad(
    op: CircuitOp,
    target_dims: tuple[int, ...],
    positions: tuple[int, ...],
    width: int,
    *,
    as_superop: bool,
) -> Array:
    """Embed *op* into a merge group, optionally lift it, and pad it to ``width``.

    :func:`~quax.embed` places ``op`` — whose qudits map to ``positions`` within the group —
    into the group's Hilbert space ``target_dims``.  The lift to a superoperator happens
    *after* embedding, which is legitimate because ``to_superop`` is a homomorphism, and which
    lets the density-matrix path reuse this whole helper unchanged.

    The trailing pad squares every group's matrix up to the stack's common width so they can
    live in one array; it has no quax equivalent because it is not a quantum operation.

    This is traceable, so it serves both the eager constant path and the vmapped parametric
    one.

    :param op: The operator to embed.
    :param target_dims: Per-qudit dimensions of the merge group.
    :param positions: Where the operator's qudits sit within the group.
    :param width: The common stack width to pad to.
    :param as_superop: Lift to a superoperator after embedding.
    :return: The padded matrix, shape ``(width, width)``.
    """
    embedded: Any = embed(op, target_dims=target_dims, positions=positions)
    if as_superop:
        embedded = to_superop(embedded)
    matrix = embedded.matrix
    return jnp.pad(matrix, [(0, width - size) for size in matrix.shape])


@final
@dataclass(kw_only=True)
class _GateBatch:
    """Gate calls sharing a constructor, constant arguments and embedding.

    Members differ only in which parameter slots feed their free arguments, so the whole batch
    is built with one :func:`jax.vmap`.  Mutable and unexported: this is a builder, filled
    during planning and then read once.
    """

    gate_fn: Callable[..., Unitary]
    num_arguments: int
    #: ``(position, value)`` for each compile-time-constant argument.
    constants: tuple[tuple[int, float], ...]
    #: Per-qudit dimensions of the merge group members embed into.
    target_dims: tuple[int, ...]
    #: Positions within the group occupied by the gate's qudits.
    group_positions: tuple[int, ...]
    #: The common stack width every embedded matrix is padded to.
    width: int
    #: Whether members are lifted to superoperators after embedding.
    as_superop: bool
    #: Stack rows this batch fills, one per member.
    rows: list[int] = field(default_factory=list)
    #: Parameter slots for each member's free arguments.
    slots: list[list[int]] = field(default_factory=list)

    def builder(self) -> Callable[[Array], Array]:
        """Return ``params -> (num_members, width, width)`` embedded matrices."""
        constant_positions = {position for position, _ in self.constants}
        free_positions = [i for i in range(self.num_arguments) if i not in constant_positions]
        gate_fn, num_arguments, constants = self.gate_fn, self.num_arguments, self.constants
        target_dims, group_positions = self.target_dims, self.group_positions
        width, as_superop = self.width, self.as_superop
        # NumPy, not JAX: this is a static index table. A jnp array built here would belong to
        # whichever trace happened to be active, and the closure outlives that trace.
        slots = np.asarray(self.slots, dtype=np.int32).reshape(len(self.rows), len(free_positions))

        def single(free_values: Array) -> Array:
            arguments: list[Any] = [None] * num_arguments
            for position, value in constants:
                arguments[position] = value
            for k, position in enumerate(free_positions):
                arguments[position] = free_values[k]
            gate = gate_fn(*arguments)
            return _embed_and_pad(gate, target_dims, group_positions, width, as_superop=as_superop)

        batched = jax.vmap(single)
        return lambda params: batched(params[slots])


def _group_fold(group_start: Sequence[int], num_ops: int, width: int) -> Callable[[Array], Array]:
    """Build the per-group matrix-product fold.

    ``fold(rows)`` takes ``(num_ops, width, width)`` embedded matrices laid out in group order
    and returns ``(num_groups, width, width)``: the ordered product of each group's members.
    Groups are gathered into a padded ``(num_groups, max_size, ...)`` array — short groups
    filled with an identity sentinel — so every group folds under one :func:`jax.vmap`.

    Representation-agnostic: composing superoperators is the same left-multiplication as
    composing unitaries, so only ``width`` differs between the two simulators.

    :param group_start: Offsets slicing the flat layout into groups.
    :param num_ops: The number of operations, i.e. the identity sentinel's row.
    :param width: The stack width.
    :return: The fold.
    """
    num_groups = len(group_start) - 1
    sizes = np.diff(np.asarray(group_start))
    max_size = int(sizes.max()) if num_groups else 1

    gather = np.full((num_groups, max_size), num_ops, dtype=np.int32)
    for group in range(num_groups):
        gather[group, : sizes[group]] = np.arange(group_start[group], group_start[group + 1])
    gather_array = gather  # static index table; see the note in _GateBatch.builder
    identity = jnp.eye(width, dtype=complex)

    def product(matrices: Array) -> Array:
        folded, _ = jax.lax.scan(lambda accumulated, m: (m @ accumulated, None), identity, matrices)
        return folded

    def fold(rows: Array) -> Array:
        padded = jnp.concatenate([rows, identity[None]], axis=0)[gather_array]
        return jax.vmap(product)(padded)

    return fold


def _stack_builder(
    circuit: ParametricCircuit,
    plan: MergePlan,
    *,
    as_superop: bool,
    d_max: int,
) -> Callable[[Array], Array]:
    """Build ``params -> (num_groups, width, width)``: the merged operator stack.

    Equal to embedding and composing each merge group of ``plan.apply(circuit.bind(params))``,
    but assembled so that the traced graph scales with the number of distinct gate *kinds*
    rather than the number of gates.  Every operation is embedded into its group's Hilbert
    space — parametric gates in vmapped batches, concrete operators eagerly, outside the
    traced graph — and each group's members are then folded together.

    :param circuit: The circuit whose operations to build.
    :param plan: The merge plan; supplies the group layout.
    :param as_superop: Lift every operation to a superoperator.
    :param d_max: The largest group Hilbert-space dimension.
    :return: The builder.
    """
    num_ops = circuit.num_ops
    width = d_max * d_max if as_superop else d_max
    dims = circuit.dims

    # Which merge group each row of the flat layout belongs to.
    group_of_row: list[tuple[int, ...]] = []
    for nodes, subsystem in plan.groups:
        group_of_row.extend([subsystem] * len(nodes))

    batches: dict[tuple[object, ...], _GateBatch] = {}
    constant_rows: list[int] = []
    constant_matrices: list[Array] = []

    for row, op_index in enumerate(plan.flat_order):
        op: ParametricOp = circuit.ops[op_index][0]
        op_subsystem = circuit.subsystems[op_index]
        group_subsystem = group_of_row[row]
        target_dims = tuple(dims[q] for q in group_subsystem)
        group_positions = tuple(group_subsystem.index(q) for q in op_subsystem)

        if isinstance(op, GateCall):
            # Keyed by embedding *shape*, not by physical qudits: two placements that trace to
            # the same graph share one vmap.
            embedding_key = (tuple(dims[q] for q in op_subsystem), target_dims, group_positions)
            constants = tuple(
                (position, value) for position, value in enumerate(op.concrete_values) if value is not None
            )
            key = op.batch_key + (embedding_key,)
            batch = batches.get(key)
            if batch is None:
                batch = _GateBatch(
                    gate_fn=op.gate_fn,
                    num_arguments=op.num_arguments,
                    constants=constants,
                    target_dims=target_dims,
                    group_positions=group_positions,
                    width=width,
                    as_superop=as_superop,
                )
                batches[key] = batch
            batch.rows.append(row)
            batch.slots.append(list(op.free_slots))
        else:
            # Concrete operations are embedded once, eagerly.  In superoperator mode this also
            # covers the non-unitary operations a noise model contributes: ``embed`` handles
            # them and ``to_superop`` is idempotent, so they need no special case.
            constant_rows.append(row)
            constant_matrices.append(_embed_and_pad(op, target_dims, group_positions, width, as_superop=as_superop))

    builders = [(np.asarray(batch.rows), batch.builder()) for batch in batches.values()]
    constant_row_array = np.asarray(constant_rows) if constant_rows else None
    constant_stack = jnp.stack(constant_matrices) if constant_matrices else None

    fold = _group_fold(plan.group_start, num_ops, width)

    def build(params: Array) -> Array:
        rows = jnp.zeros((num_ops, width, width), dtype=complex)
        for row_indices, builder in builders:
            rows = rows.at[row_indices].set(builder(params))
        if constant_stack is not None:
            rows = rows.at[constant_row_array].set(constant_stack)
        return fold(rows)

    return build


# ══════════════════════════════════════════════════════════
# Simulators
# ══════════════════════════════════════════════════════════


@dataclass(frozen=True, kw_only=True, eq=False)
class Simulator(ABC):
    """Shared evolution machinery for the differentiable simulators.

    A simulator is a *prepared* circuit: constructing one plans the merge, enumerates the
    distinct subsystems and assembles the stack builder, all of which depend only on the
    circuit's structure and not on any parameter value.  :meth:`compute` then binds
    parameters and evolves.  Preparation is therefore paid once and reused across every
    parameter vector, which is the whole reason the type exists rather than a function.

    ``eq=False`` is deliberate.  A frozen dataclass with ``eq=True`` synthesises ``__hash__``
    from its fields, and these fields hold JAX arrays, which are unhashable; identity
    semantics keep the object usable as a dictionary key and as a ``jax.jit`` static argument.

    :param circuit: The circuit to simulate.
    :param max_subsystem_size: The largest number of qudits the planner may fuse operations
        onto.  Larger values give fewer, bigger operators: usually faster to run and slower to
        compile.  Purely a performance knob — results do not depend on it.
    """

    circuit: ParametricCircuit
    max_subsystem_size: int = 2

    def __post_init__(self) -> None:
        if self.max_subsystem_size < 0:
            raise ValueError(f"max_subsystem_size must be non-negative, got {self.max_subsystem_size}.")
        self._validate_circuit()
        self.prepare()

    # ----- subclass contract -----

    @abstractmethod
    def _validate_circuit(self) -> None:
        """Reject any operation this simulator cannot evolve.  Called at construction."""

    @property
    @abstractmethod
    def _as_superop(self) -> bool:
        """Whether operations are lifted to superoperators."""

    @property
    @abstractmethod
    def initial_state(self) -> State:
        """The state evolution starts from: all qudits in :math:`|0\\rangle`."""

    @abstractmethod
    def _branch(self, subsystem: tuple[int, ...], subsystem_dims: tuple[int, ...], dim: int) -> Callable[..., State]:
        """Return the switch branch that applies an operator on one subsystem."""

    @abstractmethod
    def compute(self, params: Array | None = None) -> State:
        """Evolve :attr:`initial_state` under the circuit.

        :param params: The flat parameter vector; omit for a parameter-free circuit.
        :return: The final state.
        """

    # ----- preparation -----

    @final
    @cached_property
    def prepared_circuit(self) -> ParametricCircuit:
        """The circuit as actually simulated, after any representation change."""
        return self.circuit

    @final
    @cached_property
    def plan(self) -> MergePlan:
        """The merge plan for the prepared circuit."""
        return MergePlan.greedy(self.prepared_circuit.subsystems, self.max_subsystem_size)

    @final
    @cached_property
    def bases(self) -> tuple[tuple[int, ...], ...]:
        """The distinct subsystems the merged operations act on, in first-seen order."""
        return self.plan.bases

    @final
    @cached_property
    def base_dims(self) -> tuple[tuple[int, ...], ...]:
        """Per-qudit dimensions of each base subsystem."""
        return tuple(tuple(self.prepared_circuit.dims[q] for q in base) for base in self.bases)

    @final
    @cached_property
    def d_max(self) -> int:
        """The largest base Hilbert-space dimension; the stack's unpadded width."""
        return max((math.prod(dims) for dims in self.base_dims), default=1)

    @final
    @cached_property
    def _branches(self) -> tuple[Callable[..., State], ...]:
        return tuple(
            self._branch(base, dims, math.prod(dims)) for base, dims in zip(self.bases, self.base_dims, strict=True)
        )

    @final
    @cached_property
    def _base_index(self) -> np.ndarray:
        """Which switch branch each merged operation dispatches to.

        NumPy rather than JAX: a static index table built here must not belong to whatever
        trace happens to be active when the simulator is first used.
        """
        return np.asarray(self.plan.op_index, dtype=np.int32)

    @final
    @cached_property
    def _build_stack(self) -> Callable[[Array], Array]:
        return _stack_builder(
            self.prepared_circuit,
            self.plan,
            as_superop=self._as_superop,
            d_max=self.d_max,
        )

    @final
    @cached_property
    def _constant_stack(self) -> Array | None:
        """The whole stack, materialised eagerly when no operation takes a parameter.

        A parameter-free circuit has a stack that is a compile-time constant, so building it
        outside the traced graph spares XLA a large ``compose``/``embed`` subgraph to constant
        fold — the dominant compile cost for deep circuits of literal-angle gates.
        """
        if self.prepared_circuit.num_params:
            return None
        return self._build_stack(jnp.zeros((0,), dtype=float))

    @final
    def prepare(self) -> None:
        """Force every derived quantity, so nothing is built lazily later.

        Called at construction.  The preparation is pure structure — a merge plan, index
        tables, a stack builder — but building it lazily would let the first :meth:`compute`
        do it *inside* a JAX trace, and whatever got cached there would hold values belonging
        to a trace that has since ended.  The next call, under a different trace or none,
        would then fail with a leaked-tracer error far from the cause.  Doing it up front also
        makes the cost of constructing a simulator honest.
        """
        _ = self.prepared_circuit, self.plan, self.bases, self.base_dims, self.d_max
        _ = self.initial_state, self._branches, self._base_index
        _ = self._build_stack, self._constant_stack

    # ----- evolution -----

    @final
    def operator_stack(self, params: Array | None = None) -> Array:
        """Build the merged operator stack, one padded matrix per merge group.

        :param params: The flat parameter vector; omit for a parameter-free circuit.
        :return: Shape ``(num_groups, width, width)``, ordered as :attr:`plan` emits groups.
        """
        validated = self.prepared_circuit.validate_params(params)
        if self._constant_stack is not None:
            return self._constant_stack
        return self._build_stack(validated)

    @final
    def evolve(self, state: State, op_stack: Array) -> Any:
        """Apply a stack of operator matrices to *state*.

        A :func:`jax.lax.scan` walks the stack, dispatching each operator to the
        :func:`jax.lax.switch` branch for its subsystem, so the compiled graph grows with the
        number of distinct subsystems rather than the number of operations.

        :param state: The state to evolve.
        :param op_stack: One padded matrix per merged operation, in application order.
        :return: The evolved state.
        """
        branches = self._branches

        def body(state: Any, xs: tuple[Array, Array]) -> tuple[Any, None]:
            matrix, base = xs
            return jax.lax.switch(base, branches, matrix, state), None

        # Converted at the call site rather than cached: a fresh constant per trace is
        # correct, whereas a cached JAX array could belong to a trace that has ended.
        evolved, _ = jax.lax.scan(body, state, (op_stack, jnp.asarray(self._base_index)))
        return evolved

    @final
    def __call__(self, params: Array | None = None) -> State:
        """``simulator(params)`` is :meth:`compute`."""
        return self.compute(params)


@final
@dataclass(frozen=True, kw_only=True, eq=False)
class StateVectorSimulator(Simulator):
    """Evolve a pure state under a unitary-only circuit.

    Rejects anything that is not a unitary — channels, instruments, resets — because a state
    vector cannot represent the result.  Use :class:`DensityMatrixSimulator` for those.
    """

    @override
    def _validate_circuit(self) -> None:
        for index, (op, subsystem) in enumerate(self.circuit.ops):
            if isinstance(op, (GateCall, Unitary)):
                continue
            raise CircuitTypeError(
                f"{type(self).__name__} evolves unitary operations only, but operation {index} "
                f"on qudit(s) {list(subsystem)} is a {type(op).__name__}. Channels, "
                "instruments and resets require DensityMatrixSimulator.",
                kind=CircuitErrorKind.NON_UNITARY_OP,
                op_index=index,
                subsystem=subsystem,
            )

    @property
    @override
    def _as_superop(self) -> bool:
        return False

    @cached_property
    @override
    def initial_state(self) -> StateVector:
        """The all-zero state vector over the circuit's register."""
        return zero_state_vector(dims=self.circuit.dims)

    @override
    def _branch(
        self, subsystem: tuple[int, ...], subsystem_dims: tuple[int, ...], dim: int
    ) -> Callable[[Array, StateVector], StateVector]:
        def branch(matrix: Array, psi: StateVector) -> StateVector:
            unitary = Unitary.from_matrix(matrix[:dim, :dim], (subsystem_dims, subsystem_dims))
            return targeted_apply_unitary(unitary, psi, subsystem)

        return branch

    @override
    def compute(self, params: Array | None = None) -> StateVector:
        """Evolve the all-zero state under the circuit.

        :param params: The flat parameter vector; omit for a parameter-free circuit.
        :return: The final state vector.
        """
        if not self.plan.num_groups:
            self.prepared_circuit.validate_params(params)
            return self.initial_state
        return self.evolve(self.initial_state, self.operator_stack(params))

    def unitary(self, params: Array | None = None) -> Unitary:
        """Compose the whole circuit into a single unitary on the full register.

        Exponentially expensive in the register size; intended for verification and analysis
        rather than simulation.

        :param params: The flat parameter vector; omit for a parameter-free circuit.
        :return: The circuit's unitary.
        """
        bound = self.circuit.bind(params)
        if not bound.num_ops:
            dim = bound.dim
            return Unitary.from_matrix(jnp.eye(dim, dtype=complex), (bound.dims, bound.dims))
        composed = self.plan.apply(bound).compose()
        if not isinstance(composed, Unitary):  # pragma: no cover - guarded by _validate_circuit
            raise CircuitTypeError(
                f"Composing this circuit gave a {type(composed).__name__}, not a Unitary.",
                kind=CircuitErrorKind.NON_UNITARY_OP,
            )
        return composed


@final
@dataclass(frozen=True, kw_only=True, eq=False)
class DensityMatrixSimulator(Simulator):
    """Evolve a density matrix under any circuit, with or without noise.

    Every operation is lifted to a superoperator.  A :class:`~quax.QuantumInstrument` is
    replaced by its total channel, so a measurement acts as a dephasing channel: the result is
    the correct state averaged over outcomes, but the outcomes themselves are not recorded.
    Sampling outcomes needs a trajectory simulator, not this one.
    """

    @override
    def _validate_circuit(self) -> None:
        """Nothing to reject: every operation lifts to a superoperator."""

    @property
    @override
    def _as_superop(self) -> bool:
        return True

    @final
    @cached_property
    @override
    def prepared_circuit(self) -> ParametricCircuit:
        """The circuit with instruments collapsed to their total channels.

        Collapsing before planning is what lets a measurement merge with its neighbours: an
        instrument cannot be fused, so leaving it in place would force it to stand alone.
        """
        return self.circuit.collapse_instruments()

    @cached_property
    @override
    def initial_state(self) -> DensityMatrix:
        """The all-zero density matrix over the circuit's register."""
        return zero_state_matrix(dims=self.circuit.dims)

    @override
    def _branch(
        self, subsystem: tuple[int, ...], subsystem_dims: tuple[int, ...], dim: int
    ) -> Callable[[Array, DensityMatrix], DensityMatrix]:
        size = dim * dim

        def branch(matrix: Array, rho: DensityMatrix) -> DensityMatrix:
            superop = SuperOp.from_matrix(matrix[:size, :size], (subsystem_dims, subsystem_dims))
            return targeted_apply_superop(superop, rho, subsystem)

        return branch

    @override
    def compute(self, params: Array | None = None) -> DensityMatrix:
        """Evolve the all-zero density matrix under the circuit.

        :param params: The flat parameter vector; omit for a parameter-free circuit.
        :return: The final density matrix.
        """
        if not self.plan.num_groups:
            self.prepared_circuit.validate_params(params)
            return self.initial_state
        return self.evolve(self.initial_state, self.operator_stack(params))


def simulate(circuit: Circuit | ParametricCircuit, params: Array | None = None) -> State:
    """Simulate *circuit* once, choosing the representation from its contents.

    A convenience for one-off use: a unitary-only circuit is evolved as a state vector, and
    anything else as a density matrix.  Constructing a simulator directly is better whenever
    the circuit is simulated more than once, since that reuses the preparation.

    :param circuit: The circuit to simulate.
    :param params: The flat parameter vector; omit for a parameter-free circuit.
    :return: A ``StateVector`` for a unitary-only circuit, otherwise a ``DensityMatrix``.
    """
    parametric = circuit if isinstance(circuit, ParametricCircuit) else ParametricCircuit.from_circuit(circuit)
    unitary_only = all(isinstance(op, (GateCall, Unitary)) for op, _ in parametric.ops)
    simulator: Simulator = (
        StateVectorSimulator(circuit=parametric) if unitary_only else DensityMatrixSimulator(circuit=parametric)
    )
    return simulator.compute(params)
