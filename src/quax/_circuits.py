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

"""Straight-line circuits over a qudit register, and structural merge planning.

A :class:`Circuit` is an ordered sequence of operations, each placed on a subsystem of a fixed
qudit register.  An operation is either an operator that is already built or a
:class:`ParameterizedGate` whose arguments arrive later in a flat parameter vector.  It is
deliberately not a program: there are no gate names, no classical memory and no control flow.
Building one is the job of whatever owns the source language — a Quil compiler, say — and this
module's job begins once the operations exist.

A :class:`ConstantCircuit` is the narrower case in which every operation is already built.  It
is what the operator algebra consumes, because composing and merging need matrices.  The two
convert both ways: :meth:`Circuit.to_constant_circuit` builds every gate, and
:meth:`ConstantCircuit.to_circuit` widens a built circuit into one that takes no parameters.

A :class:`MergePlan` is the purely combinatorial half of operator fusion: given only the
subsystems each operation acts on, it decides which operations may be merged into a single
larger operator without reordering anything that does not commute.  A plan is data, not a
closure: its ``groups``, ``bases`` and ``op_index`` are the inputs a simulator needs in order
to build a fused operator stack *without* first materialising every operator, which is what
makes vectorised (``jax.vmap``) stack construction possible for parameterized circuits.

    >>> circuit = ConstantCircuit.from_ops([(gates.H, (0,)), (gates.CNOT, (0, 1)), (gates.X, (1,))])
    >>> plan = MergePlan.greedy(circuit.subsystems, max_subsystem_size=2)
    >>> plan.apply(circuit).num_ops
    1
"""

import heapq
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property, reduce
from operator import mul
from typing import TypeAlias, final

import jax
import jax.numpy as jnp
from jax import Array

from ._errors import CircuitErrorKind, CircuitTypeError, CircuitValueError
from ._promotion import embed
from ._quantum_objects import (
    KrausMap,
    QuantumInstrument,
    SuperOperator,
    Unitary,
)
from ._random import random_choi, random_unitary
from ._superoperator_transformations import to_kraus, to_superop, truncate_kraus

#: An operator that is already built.  ``Unitary`` covers ``Involution`` (and hence the constant
#: gates); ``SuperOperator`` covers ``SuperOp``, ``KrausMap``, ``Choi`` and ``PauliLiouville``.
#: ``Lindbladian`` is excluded: it is a generator, not an operation.
#:
#: The qualified name is deliberate.  :data:`CircuitOp` is the general thing an operation may
#: be — built or not — and this is the narrower case, so this one carries the qualifier.
ConstantOp: TypeAlias = Unitary | SuperOperator | QuantumInstrument

#: One already-built operation: an operator together with the register indices it acts on, in
#: operand order.
ConstantPlacement: TypeAlias = tuple[ConstantOp, tuple[int, ...]]

#: A group in a merge plan: the operation indices it fuses, and the subsystem it acts on.
Group: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]


def dependency_edges(subsystems: Sequence[tuple[int, ...]]) -> tuple[tuple[int, int], ...]:
    """Return the dependency edges induced by a sequence of subsystems.

    An edge ``(u, v)`` means operation ``u`` must be applied before operation ``v`` because
    they share a qudit and ``u`` comes first.  Only the *immediate* predecessor on each qudit
    is recorded; the transitive closure is implied.

    Because every edge runs from a lower index to a higher one, the identity permutation
    ``0, 1, ..., n - 1`` is always a valid topological order of the result.

    :param subsystems: One tuple of register indices per operation, in application order.
    :return: Edges as ``(predecessor, successor)`` pairs.
    """
    edges: list[tuple[int, int]] = []
    last_on_qudit: dict[int, int] = {}
    for index, subsystem in enumerate(subsystems):
        for qudit in subsystem:
            previous = last_on_qudit.get(qudit)
            if previous is not None:
                edges.append((previous, index))
            last_on_qudit[qudit] = index
    return tuple(edges)


def validate_placement_indices(
    index: int,
    subsystem: tuple[int, ...],
    num_qudits: int,
    dims: tuple[int, ...],
) -> None:
    """Check that one operation's subsystem indexes the register legally.

    Shared by :class:`ConstantCircuit` and :class:`Circuit`.  Only the *indices* are
    checked, never the operator's dimensions: a circuit's gates may be unbuilt, so
    their dimensions are not knowable until parameters are bound.

    :param index: Position of the operation, for the error message.
    :param subsystem: Register indices the operation acts on.
    :param num_qudits: Size of the register.
    :param dims: Per-qudit dimensions, for the error message.
    :raises CircuitValueError: If an index is out of range or repeated.
    """
    out_of_range = [q for q in subsystem if not 0 <= q < num_qudits]
    if out_of_range:
        raise CircuitValueError(
            f"Operation {index} acts on qudit(s) {out_of_range}, outside a register of "
            f"{num_qudits} qudit(s) with dims={dims}.",
            kind=CircuitErrorKind.SUBSYSTEM_OUT_OF_RANGE,
            op_index=index,
            subsystem=subsystem,
        )
    if len(set(subsystem)) != len(subsystem):
        raise CircuitValueError(
            f"Operation {index} names a qudit more than once: {subsystem}.",
            kind=CircuitErrorKind.DUPLICATE_QUDIT,
            op_index=index,
            subsystem=subsystem,
        )


@final
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, kw_only=True)
class ConstantCircuit:
    """A circuit in which every operation is already built.

    The narrower of the two circuit types: :class:`Circuit` is the general case, and this is
    what remains once :meth:`Circuit.to_constant_circuit` has supplied every gate's arguments.
    Composition, merging and representation changes live here rather than on ``Circuit``,
    because each of them needs matrices.

    A circuit describes one *straight-line block*: the operations are applied in order, with
    no branching.  Anything that requires branching on a measurement outcome is the caller's
    concern, and is naturally expressed as several circuits.

    Operators are stored exactly as given, including their operand order, so a ``CNOT`` placed
    on ``(1, 0)`` keeps its control and target the way the caller wrote them.

    The register's dimensions are an upper bound, not an exact match: an operator may act on a
    subsystem whose register dimension is *larger* than its own, in which case
    :func:`quax.embed` promotes it when needed.  This is what lets a qubit gate sit in a
    qutrit register.

    :param dims: Per-qudit dimensions of the register, e.g. ``(2, 2, 3)``.
    :param ops: The operations, each an ``(operator, subsystem)`` pair, in application order.
    """

    dims: tuple[int, ...]
    ops: tuple[ConstantPlacement, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(int(d) for d in self.dims))
        object.__setattr__(self, "ops", tuple((op, tuple(int(q) for q in sub)) for op, sub in self.ops))

        if any(d < 1 for d in self.dims):
            raise ValueError(f"Register dimensions must be positive, got {self.dims}.")

        num_qudits = len(self.dims)
        for index, (op, subsystem) in enumerate(self.ops):
            validate_placement_indices(index, subsystem, num_qudits, self.dims)

            op_dims = op.dims[1]
            if len(op_dims) != len(subsystem):
                raise CircuitValueError(
                    f"Operation {index} acts on {len(op_dims)} qudit(s) but is placed on "
                    f"{len(subsystem)} register position(s) {subsystem}.",
                    kind=CircuitErrorKind.OPERATOR_ARITY_MISMATCH,
                    op_index=index,
                    subsystem=subsystem,
                )
            too_large = [(q, d, self.dims[q]) for q, d in zip(subsystem, op_dims, strict=True) if d > self.dims[q]]
            if too_large:
                detail = ", ".join(f"qudit {q}: operator dim {d} > register dim {rd}" for q, d, rd in too_large)
                raise CircuitValueError(
                    f"Operation {index} does not fit the register ({detail}).",
                    kind=CircuitErrorKind.OPERATOR_EXCEEDS_REGISTER,
                    op_index=index,
                    subsystem=subsystem,
                )

    # ----- pytree -----

    def tree_flatten(self) -> tuple[tuple[ConstantOp, ...], tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]]:
        return self.operators, (self.dims, self.subsystems)

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: Iterable[ConstantOp]) -> "ConstantCircuit":
        dims, subsystems = aux_data
        # Bypass ``__post_init__``: unflattening happens with traced (and sometimes
        # placeholder) children, whose ``dims`` are not meaningful to validate.
        circuit = object.__new__(cls)
        object.__setattr__(circuit, "dims", dims)
        object.__setattr__(circuit, "ops", tuple(zip(children, subsystems, strict=True)))
        return circuit

    # ----- display -----

    def __str__(self) -> str:
        return f"ConstantCircuit(dims={self.dims}, num_ops={self.num_ops})"

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self) -> Iterator[ConstantPlacement]:
        return iter(self.ops)

    def __getitem__(self, index: int) -> ConstantPlacement:
        """Return one operation as an ``(operator, subsystem)`` pair.

        Slicing is not supported: a slice of a circuit would still carry the whole register,
        which is rarely what a caller means.  Build one explicitly with :meth:`with_ops`.
        """
        return self.ops[index]

    # ----- structure -----

    @property
    def num_ops(self) -> int:
        """The number of operations."""
        return len(self.ops)

    @property
    def num_qudits(self) -> int:
        """The number of qudits in the register."""
        return len(self.dims)

    @property
    def dim(self) -> int:
        """The total Hilbert-space dimension of the register."""
        return reduce(mul, self.dims, 1)

    @cached_property
    def subsystems(self) -> tuple[tuple[int, ...], ...]:
        """The register indices each operation acts on, in operand order."""
        return tuple(subsystem for _, subsystem in self.ops)

    @cached_property
    def operators(self) -> tuple[ConstantOp, ...]:
        """The operators, in application order."""
        return tuple(op for op, _ in self.ops)

    # ----- construction -----

    @staticmethod
    def infer_dims(
        ops: Sequence[ConstantPlacement],
        num_qudits: int | None = None,
        default_dim: int = 2,
    ) -> tuple[int, ...]:
        """Infer register dimensions as the largest dimension each qudit is acted on with.

        ``default_dim`` applies only to qudits that no operation touches: a qudit an operation
        acts on takes its dimension from the operators, so a qubit gate does not silently
        widen its slot.  Pass ``dims`` to :class:`ConstantCircuit` directly to size a register the
        operators do not determine — an all-qutrit register holding qubit gates, say.

        :param ops: The operations, as ``(operator, subsystem)`` pairs.
        :param num_qudits: Register size.  Defaults to one past the largest index used.
        :param default_dim: Dimension for a qudit no operation touches.
        :return: Per-qudit dimensions.
        """
        if num_qudits is None:
            num_qudits = 1 + max((q for _, sub in ops for q in sub), default=-1)
        dims = [0] * num_qudits
        for op, subsystem in ops:
            # Not strict: a subsystem/operator arity mismatch is reported by ``__post_init__``
            # with the operation index and a usable message, which a zip failure here would
            # pre-empt with an opaque one.
            for qudit, d in zip(subsystem, op.dims[1]):
                dims[qudit] = max(dims[qudit], d)
        return tuple(d or default_dim for d in dims)

    @classmethod
    def from_ops(
        cls,
        ops: Sequence[ConstantPlacement],
        num_qudits: int | None = None,
        default_dim: int = 2,
    ) -> "ConstantCircuit":
        """Build a circuit, inferring the register dimensions from the operators.

        :param ops: The operations, as ``(operator, subsystem)`` pairs, in application order.
        :param num_qudits: Register size.  Defaults to one past the largest index used.
        :param default_dim: Dimension for a qudit no operation touches.
        :return: The circuit.
        """
        return cls(dims=cls.infer_dims(ops, num_qudits, default_dim), ops=tuple(ops))

    def with_ops(self, ops: Sequence[ConstantPlacement]) -> "ConstantCircuit":
        """Return a circuit with the same register and different operations."""
        return ConstantCircuit(dims=self.dims, ops=tuple(ops))

    def to_circuit(self) -> "Circuit":
        """Widen to a :class:`Circuit` that takes no parameters.

        Every already-built circuit is trivially a circuit whose gates happen to need nothing
        bound.  This is the inverse of :meth:`Circuit.to_constant_circuit`, and the way to hand
        an already-built circuit to something that consumes the general type — a simulator, say.

        :return: An equivalent :class:`Circuit` with ``num_params == 0``.
        """
        return Circuit(dims=self.dims, ops=tuple(self.ops), num_params=0)

    # ----- representation changes -----

    def to_superops(self) -> "ConstantCircuit":
        """Convert every operation to a :class:`~quax.SuperOp`.

        Unitaries, Kraus maps, Choi matrices and Pauli-Liouville matrices are converted
        directly.  A :class:`~quax.QuantumInstrument` is replaced by its total channel, which
        discards the outcome labels — the resulting circuit describes the unconditioned
        evolution, which is what density-matrix evolution needs.

        :return: A circuit whose operators are all ``SuperOp``.
        """
        converted: list[ConstantPlacement] = []
        for op, subsystem in self.ops:
            channel = op.total_channel() if isinstance(op, QuantumInstrument) else op
            converted.append((to_superop(channel), subsystem))
        return self.with_ops(converted)

    def to_kraus_maps(self, atol: float = 1e-6) -> "ConstantCircuit":
        """Convert channels to truncated :class:`~quax.KrausMap` operators.

        ``SuperOp``, ``Choi`` and ``PauliLiouville`` operations become ``KrausMap`` operations
        with negligible Kraus operators dropped.  ``Unitary``, ``KrausMap`` and
        ``QuantumInstrument`` operations pass through unchanged: each is already applicable to
        a state vector, deterministically or by sampling, so lifting them would only cost
        precision and memory.

        :param atol: Kraus operators with smaller norm are discarded.
        :return: A circuit with no dense superoperators.
        """
        converted: list[ConstantPlacement] = []
        for op, subsystem in self.ops:
            if isinstance(op, SuperOperator) and not isinstance(op, KrausMap):
                converted.append((truncate_kraus(to_kraus(op), atol=atol), subsystem))
            else:
                converted.append((op, subsystem))
        return self.with_ops(converted)

    # ----- algebra -----

    def compose(self) -> ConstantOp:
        """Fold the whole circuit into a single operator on the full register.

        Each operation is embedded into the register's Hilbert space and composed in
        application order.  The result is a ``Unitary`` when every operation is unitary and a
        superoperator as soon as one is not.

        This is exponentially expensive in the register size and is intended for verification
        and analysis, not for simulation.

        :return: The composed operator, acting on all ``num_qudits`` qudits.
        :raises ValueError: If the circuit is empty.
        :raises TypeError: If any operation is a ``QuantumInstrument``, which has no
            composition — call :meth:`to_superops` first to compose the total channel.
        """
        if not self.ops:
            raise ValueError("Cannot compose an empty circuit.")
        instruments = [i for i, (op, _) in enumerate(self.ops) if isinstance(op, QuantumInstrument)]
        if instruments:
            raise CircuitTypeError(
                f"Operation(s) {instruments} are QuantumInstruments, which do not compose. "
                "Call to_superops() first to compose their total channels instead.",
                kind=CircuitErrorKind.INSTRUMENT_IN_MERGE_GROUP,
                op_index=instruments[0],
            )
        return _merge(self.ops, tuple(range(self.num_qudits)), self.dims)


def _merge(
    ops: Sequence[ConstantPlacement],
    subsystem: tuple[int, ...],
    dims: tuple[int, ...],
) -> ConstantOp:
    """Embed each operation into ``subsystem`` and compose them in application order.

    ``@`` promotes mixed operator types, so an all-unitary group folds to a ``Unitary`` while
    a group containing any channel folds to a superoperator.

    :param ops: The operations to merge, in application order.
    :param subsystem: Register indices of the merged operator, ascending.
    :param dims: Per-qudit dimensions of the whole register.
    :return: The composed operator, acting on ``subsystem``.
    """
    target_dims = tuple(dims[q] for q in subsystem)
    accumulated: ConstantOp | None = None
    for op, op_subsystem in ops:
        positions = tuple(subsystem.index(q) for q in op_subsystem)
        embedded = embed(op, target_dims=target_dims, positions=positions)
        accumulated = embedded if accumulated is None else embedded @ accumulated
    if accumulated is None:
        raise ValueError("Cannot merge an empty operation group.")
    return accumulated


@final
class _UnionFind:
    """Disjoint-set forest with union by rank and path compression."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, x: int) -> int:
        """Return the representative of ``x``'s set, compressing the path to it."""
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x: int, y: int) -> int:
        """Merge the sets containing ``x`` and ``y`` and return the new representative."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return root_x
        if self._rank[root_x] < self._rank[root_y]:
            root_x, root_y = root_y, root_x
        self._parent[root_y] = root_x
        if self._rank[root_x] == self._rank[root_y]:
            self._rank[root_x] += 1
        return root_x


@final
class _Quotient:
    """A mutable DAG over group representatives, contracted as groups merge.

    It starts as the operations' dependency graph and is the authority on whether a candidate
    merge is *convex*: contracting two groups must not reorder any operation that lies
    topologically between them.
    """

    def __init__(self, num_nodes: int, edges: Iterable[tuple[int, int]]) -> None:
        self.successors: dict[int, set[int]] = {n: set() for n in range(num_nodes)}
        self.predecessors: dict[int, set[int]] = {n: set() for n in range(num_nodes)}
        for u, v in edges:
            self.successors[u].add(v)
            self.predecessors[v].add(u)

    def creates_cycle(self, a: int, b: int) -> bool:
        """Return ``True`` if contracting ``a`` and ``b`` would create a cycle.

        The quotient is always a DAG, so contracting two nodes introduces a cycle iff there is
        a directed path of length two or more between them in *either* direction — some other
        group is sandwiched on a dependency path from one to the other, and merging across it
        would force it to be reordered.  A direct edge alone is fine; only an indirect path is
        a problem.
        """
        for source, target in ((a, b), (b, a)):
            stack = [s for s in self.successors[source] if s != target]
            seen = set(stack)
            while stack:
                node = stack.pop()
                if node == target:
                    return True
                for following in self.successors[node]:
                    if following not in seen:
                        seen.add(following)
                        stack.append(following)
        return False

    def contract(self, keep: int, drop: int) -> None:
        """Contract ``drop`` into ``keep``."""
        for pred in self.predecessors[drop]:
            if pred != keep:
                self.successors[pred].discard(drop)
                self.successors[pred].add(keep)
                self.predecessors[keep].add(pred)
        for succ in self.successors[drop]:
            if succ != keep:
                self.predecessors[succ].discard(drop)
                self.predecessors[succ].add(keep)
                self.successors[keep].add(succ)
        self.successors[keep].discard(drop)
        self.predecessors[keep].discard(drop)
        del self.successors[drop]
        del self.predecessors[drop]

    def topological_order(self, key: dict[int, int]) -> list[int]:
        """Return the nodes in topological order, breaking ties by smallest ``key``."""
        indegree = {n: len(preds) for n, preds in self.predecessors.items()}
        ready = [(key[n], n) for n, d in indegree.items() if d == 0]
        heapq.heapify(ready)
        order: list[int] = []
        while ready:
            _, node = heapq.heappop(ready)
            order.append(node)
            for succ in self.successors[node]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    heapq.heappush(ready, (key[succ], succ))
        if len(order) != len(indegree):
            raise RuntimeError("Quotient graph is cyclic; this is a bug in merge planning.")
        return order


@final
@dataclass(frozen=True, kw_only=True)
class MergePlan:
    """A structural recipe for fusing a circuit's operations into groups.

    A plan depends only on *which* subsystems the operations act on — never on the operators
    themselves, nor on any parameter value.  That is what makes it usable as data: a simulator
    can read the group structure, the distinct subsystems and the per-group subsystem index off
    a plan and build a fused operator stack under ``jax.vmap`` without ever materialising the
    individual operators.  Call :meth:`apply` to do the materialising merge instead.

    Groups are listed in an order that respects every dependency, and each group lists its
    members in application order.  A group of one operation keeps that operation's own operand
    order; a group of several is recorded on the ascending union of its members' subsystems,
    which is the order :meth:`apply` embeds them into.

    .. warning::
        A group's *position* is not its application order.  Two operations that share no qudit
        commute, so merging can legitimately swap them: a group blocked behind a wide merge may
        be emitted after an independent operation that came later in the original circuit.  Any
        caller that needs to label results per operation — measurement outcome columns, say —
        must use the operation indices in :attr:`groups`, never the group's position.

    :param groups: One ``(operation indices, subsystem)`` pair per group, in application order.
    :param num_ops: The number of operations the plan covers.
    """

    groups: tuple[Group, ...]
    num_ops: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "groups",
            tuple((tuple(int(n) for n in nodes), tuple(int(q) for q in sub)) for nodes, sub in self.groups),
        )
        covered = [node for nodes, _ in self.groups for node in nodes]
        if sorted(covered) != list(range(self.num_ops)):
            raise ValueError(
                f"A merge plan must cover each of the {self.num_ops} operation(s) exactly once; "
                f"got {len(covered)} entries covering {len(set(covered))} distinct operations."
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __str__(self) -> str:
        return f"MergePlan(num_ops={self.num_ops}, num_groups={self.num_groups}, num_bases={len(self.bases)})"

    @property
    def num_groups(self) -> int:
        """The number of groups, i.e. the number of operations after merging."""
        return len(self.groups)

    @cached_property
    def bases(self) -> tuple[tuple[int, ...], ...]:
        """The distinct subsystems the groups act on, in first-seen order.

        A simulator that dispatches each fused operation through a ``jax.lax.switch`` needs one
        branch per base, so the size of its compiled graph scales with the number of bases
        rather than the number of operations.
        """
        seen: dict[tuple[int, ...], int] = {}
        for _, subsystem in self.groups:
            if subsystem not in seen:
                seen[subsystem] = len(seen)
        return tuple(seen)

    @cached_property
    def op_index(self) -> tuple[int, ...]:
        """The index into :attr:`bases` of each group's subsystem, in application order."""
        lookup = {subsystem: i for i, subsystem in enumerate(self.bases)}
        return tuple(lookup[subsystem] for _, subsystem in self.groups)

    @cached_property
    def flat_order(self) -> tuple[int, ...]:
        """Operation indices laid out group by group, in application order within each group.

        This is the layout a vectorised stack builder wants: every operation appears exactly
        once, and :attr:`group_start` slices it into groups.  Reading the layout off the plan
        means the builder never has to materialise an operator to discover where it belongs.
        """
        return tuple(node for nodes, _ in self.groups for node in nodes)

    @cached_property
    def group_start(self) -> tuple[int, ...]:
        """Offsets into :attr:`flat_order`, one per group plus a final sentinel.

        Group ``g`` occupies ``flat_order[group_start[g]:group_start[g + 1]]``, so the tuple
        has ``num_groups + 1`` entries and its last entry is :attr:`num_ops`.
        """
        starts = [0]
        for nodes, _ in self.groups:
            starts.append(starts[-1] + len(nodes))
        return tuple(starts)

    @cached_property
    def max_group_size(self) -> int:
        """The largest number of operations in any one group; ``1`` when nothing merged."""
        return max((len(nodes) for nodes, _ in self.groups), default=1)

    @property
    def compression_ratio(self) -> float:
        """Groups per original operation; 1.0 when nothing was merged."""
        return self.num_groups / self.num_ops if self.num_ops else 1.0

    @classmethod
    def trivial(cls, subsystems: Sequence[tuple[int, ...]]) -> "MergePlan":
        """Return the plan that merges nothing, keeping every operation as written.

        :param subsystems: One tuple of register indices per operation, in application order.
        :return: A plan with one single-operation group per operation.
        """
        return cls(groups=tuple(((i,), tuple(sub)) for i, sub in enumerate(subsystems)), num_ops=len(subsystems))

    @classmethod
    def greedy(
        cls,
        subsystems: Sequence[tuple[int, ...]],
        max_subsystem_size: int,
        *,
        atomic: Iterable[int] = (),
    ) -> "MergePlan":
        """Plan a merge by greedy edge contraction, smallest candidate union first.

        Merging small operations into larger neighbours reduces the number of distinct
        subsystem shapes, and therefore the size of a simulator's compiled graph, more than it
        reduces the operation count.  So candidates are contracted in ascending order of the
        subsystem size they would produce, which absorbs single-qudit operations into
        multi-qudit neighbours first.

        Two groups are contracted only when the union fits within ``max_subsystem_size`` *and*
        the contraction is convex — no operation that depends on one group and is depended on
        by the other gets reordered.  Convexity is what makes merging safe in the presence of
        non-commuting operations.

        :param subsystems: One tuple of register indices per operation, in application order.
        :param max_subsystem_size: The largest number of qudits a group may span.  ``0``
            disables merging, giving :meth:`trivial`.
        :param atomic: Operations that must never be merged.  Use this for any operation a
            caller needs to interact with individually — a
            :class:`~quax.QuantumInstrument` whose outcome is sampled, for instance, is
            unobservable once fused into a neighbour, even though fusing it would preserve the
            circuit's overall channel.  An atomic operation keeps its place relative to
            everything it depends on, but not relative to independent operations; see the
            warning on :class:`MergePlan`.
        :return: The plan.
        """
        num_ops = len(subsystems)
        if max_subsystem_size <= 0 or num_ops == 0:
            return cls.trivial(subsystems)

        atomic_nodes = frozenset(int(n) for n in atomic)
        out_of_range = sorted(n for n in atomic_nodes if not 0 <= n < num_ops)
        if out_of_range:
            raise ValueError(f"atomic contains operation index(es) {out_of_range} outside 0..{num_ops - 1}.")

        edges = dependency_edges(subsystems)
        union_find = _UnionFind(num_ops)
        quotient = _Quotient(num_ops, edges)
        group_qudits: dict[int, set[int]] = {i: set(sub) for i, sub in enumerate(subsystems)}

        neighbours: dict[int, set[int]] = {n: set() for n in range(num_ops)}
        for u, v in edges:
            neighbours[u].add(v)
            neighbours[v].add(u)

        # Candidate heap keyed by the size of the union a contraction would produce.
        candidates: list[tuple[int, int, int]] = []
        for u, v in edges:
            if u in atomic_nodes or v in atomic_nodes:
                continue
            union_size = len(group_qudits[u] | group_qudits[v])
            if union_size <= max_subsystem_size:
                heapq.heappush(candidates, (union_size, u, v))

        while candidates:
            _, u, v = heapq.heappop(candidates)
            root_u, root_v = union_find.find(u), union_find.find(v)
            if root_u == root_v:
                continue
            union_qudits = group_qudits[root_u] | group_qudits[root_v]
            if len(union_qudits) > max_subsystem_size:
                continue
            if quotient.creates_cycle(root_u, root_v):
                continue

            new_root = union_find.union(root_u, root_v)
            dropped = root_v if new_root == root_u else root_u
            group_qudits[new_root] = union_qudits
            del group_qudits[dropped]
            quotient.contract(new_root, dropped)

            for neighbour in neighbours[u] | neighbours[v]:
                if neighbour in atomic_nodes:
                    continue
                root_neighbour = union_find.find(neighbour)
                if root_neighbour == new_root:
                    continue
                union_size = len(group_qudits[new_root] | group_qudits[root_neighbour])
                if union_size <= max_subsystem_size:
                    heapq.heappush(candidates, (union_size, u, neighbour))

        # Every dependency edge runs from a lower index to a higher one, so ascending operation
        # index is a topological order of the original graph: listing each group's members in
        # index order composes them in application order.
        members: dict[int, list[int]] = {}
        first_member: dict[int, int] = {}
        for node in range(num_ops):
            root = union_find.find(node)
            members.setdefault(root, []).append(node)
            first_member.setdefault(root, node)

        # Emit groups in topological order of the *quotient*, not of the original graph.  A
        # valid group may hold an operation preceding an atomic one together with an operation
        # depending on it; emitting the group at its earliest member's position would move the
        # whole group — the dependent operation included — ahead of the atomic operation.
        # Breaking ties on each group's first member keeps unmerged operations, atomic ones in
        # particular, in application order relative to each other.
        groups: list[Group] = []
        for root in quotient.topological_order(first_member):
            nodes = members[root]
            subsystem = subsystems[nodes[0]] if len(nodes) == 1 else tuple(sorted(group_qudits[root]))
            groups.append((tuple(nodes), tuple(subsystem)))
        return cls(groups=tuple(groups), num_ops=num_ops)

    def apply(self, circuit: ConstantCircuit) -> ConstantCircuit:
        """Merge a circuit's operations according to this plan.

        A single-operation group passes its operator through untouched.  A multi-operation
        group is embedded into the group's subsystem and composed, so an all-unitary group
        yields a ``Unitary`` and a group containing any channel yields a superoperator.

        :param circuit: The circuit to merge.  Its operation count must match
            :attr:`num_ops`, and its subsystems are expected to be the ones the plan was
            built from.
        :return: The merged circuit, on the same register.
        :raises ValueError: If the circuit's operation count does not match the plan.
        :raises TypeError: If a multi-operation group contains a ``QuantumInstrument``.
        """
        if circuit.num_ops != self.num_ops:
            raise ValueError(
                f"This plan covers {self.num_ops} operation(s) but the circuit has "
                f"{circuit.num_ops}. Rebuild the plan from circuit.subsystems."
            )
        merged: list[ConstantPlacement] = []
        for nodes, subsystem in self.groups:
            if len(nodes) == 1:
                merged.append(circuit.ops[nodes[0]])
                continue
            group_ops = [circuit.ops[node] for node in nodes]
            instruments = [
                node for node, (op, _) in zip(nodes, group_ops, strict=True) if isinstance(op, QuantumInstrument)
            ]
            if instruments:
                raise CircuitTypeError(
                    f"Group {nodes} contains QuantumInstrument operation(s) {instruments}, which "
                    "cannot be merged: fusing an instrument into a neighbour discards the outcome. "
                    "Pass those indices as MergePlan.greedy(..., atomic=...).",
                    kind=CircuitErrorKind.INSTRUMENT_IN_MERGE_GROUP,
                    op_index=instruments[0],
                )
            merged.append((_merge(group_ops, subsystem, circuit.dims), subsystem))
        return circuit.with_ops(merged)


def random_constant_circuit(
    dims: tuple[int, ...],
    num_ops: int,
    key: Array,
    *,
    max_arity: int = 2,
    channel_probability: float = 0.0,
    kraus_rank: int = 2,
) -> ConstantCircuit:
    """Generate a random circuit over a register.

    Each operation gets a uniformly random arity up to ``max_arity``, a uniformly random
    subsystem of that arity, and a Haar-random unitary — or, with probability
    ``channel_probability``, a BCSZ-random channel as a ``SuperOp``.

    :param dims: Per-qudit dimensions of the register.
    :param num_ops: The number of operations to generate.
    :param key: A JAX PRNG key.
    :param max_arity: The largest number of qudits one operation may act on.  Clipped to the
        register size.
    :param channel_probability: The probability that an operation is a channel rather than a
        unitary.
    :param kraus_rank: The Kraus rank of generated channels.
    :return: The circuit.
    """
    num_qudits = len(dims)
    if num_qudits == 0:
        raise ValueError("Cannot generate a circuit over an empty register.")
    max_arity = min(max_arity, num_qudits)
    if max_arity < 1:
        raise ValueError(f"max_arity must be at least 1, got {max_arity}.")

    ops: list[ConstantPlacement] = []
    for _ in range(num_ops):
        key, arity_key, subsystem_key, op_key, kind_key = jax.random.split(key, 5)
        arity = int(jax.random.randint(arity_key, (), 1, max_arity + 1))
        subsystem = tuple(int(q) for q in jax.random.choice(subsystem_key, num_qudits, (arity,), replace=False))
        op_dims = tuple(dims[q] for q in subsystem)
        if channel_probability > 0.0 and float(jax.random.uniform(kind_key)) < channel_probability:
            op: ConstantOp = to_superop(random_choi((op_dims, op_dims), rank=kraus_rank, key=op_key))
        else:
            op = random_unitary((op_dims, op_dims), key=op_key)
        ops.append((op, subsystem))
    return ConstantCircuit(dims=tuple(dims), ops=tuple(ops))


# ══════════════════════════════════════════════════════════
# Circuits with parameterized gates
# ══════════════════════════════════════════════════════════


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class Slot:
    """A gate argument supplied at run time, from one entry of the parameter vector.

    :param index: Which entry of the flat parameter vector feeds this argument.
    """

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"Parameter slot must be non-negative, got {self.index}.")

    def __str__(self) -> str:
        return f"params[{self.index}]"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class Constant:
    """A gate argument fixed when the circuit is built.

    :param value: The argument's value.
    """

    value: float

    def __str__(self) -> str:
        return repr(self.value)


#: One argument of a :class:`ParameterizedGate`: either a runtime slot or a fixed value.
#:
#: A tagged union rather than a bare ``int | float`` discriminated by runtime type.  The bare
#: version reads well but is unsafe in exactly the way this package cannot afford: ``bool`` is
#: an ``int``, an integer-valued constant such as ``0`` is indistinguishable from slot ``0``,
#: and ``typing.NewType`` is erased at runtime so it cannot rescue either case.  The failure
#: mode is a silently different circuit, which is the class of bug the slot invariant exists
#: to rule out.  These wrappers are built once per gate argument at construction and never
#: enter a traced computation, so they cost nothing where it matters.
GateArgument: TypeAlias = Slot | Constant


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ParameterizedGate:
    """A gate constructor together with the arguments it will be built from.

    A :class:`ParameterizedGate` is a gate that has not been built yet.  It exists so that a
    circuit can be *structurally* complete — every operation placed on the register, every
    dependency known — while the numbers that determine its matrices arrive later.

    The point is not laziness but batching.  Because the constructor and the argument layout
    are available as **data**, a consumer can group every gate sharing them and build the whole
    group under a single :func:`jax.vmap`.  A traced Python closure ``params -> gate`` could not
    be grouped that way: nothing can be read off it before it runs.  So the traced graph grows
    with the number of distinct gate *kinds* rather than the number of gates, which is the
    difference between a compile time that is flat in circuit depth and one that is not.

        >>> ParameterizedGate(gate_fn=gates.RX, arguments=(Slot(index=0),))
        >>> ParameterizedGate(gate_fn=gates.PHASEDRX, arguments=(Constant(value=0.5), Slot(index=3)))

    :param gate_fn: The gate constructor, e.g. ``quax.gates.RX``.  Must be hashable and stable
        across calls — it is part of :attr:`batch_key`, so a constructor created afresh at each
        call site defeats batching even when the gates are identical.
    :param arguments: One :class:`Slot` or :class:`Constant` per constructor argument, in
        order.
    """

    gate_fn: Callable[..., Unitary]
    arguments: tuple[GateArgument, ...]

    @property
    def name(self) -> str:
        """The constructor's name, for error messages."""
        return getattr(self.gate_fn, "__name__", repr(self.gate_fn))

    @property
    def num_arguments(self) -> int:
        """The number of arguments the gate takes."""
        return len(self.arguments)

    @property
    def free_slots(self) -> tuple[int, ...]:
        """The parameter slots this gate reads, in argument order."""
        return tuple(argument.index for argument in self.arguments if isinstance(argument, Slot))

    @property
    def constants(self) -> tuple[tuple[int, float], ...]:
        """``(position, value)`` for each argument fixed at build time."""
        return tuple(
            (position, argument.value)
            for position, argument in enumerate(self.arguments)
            if isinstance(argument, Constant)
        )

    @property
    def batch_key(self) -> tuple[object, ...]:
        """Identity for grouping gates that can be built under one ``jax.vmap``.

        Two gates share a key when they invoke the same constructor with the same constants
        pinned to the same argument positions — everything that shapes the traced graph.  Which
        slots the free arguments read is deliberately *not* part of the key: that becomes the
        vmapped axis.

        Keyed on the function object rather than its ``id``, which is reusable after garbage
        collection and differs per call site for a locally created constructor.
        """
        return (self.gate_fn, self.num_arguments, self.constants)

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(argument) for argument in self.arguments)})"

    def __call__(self, params: Array) -> Unitary:
        """Build the gate, reading its runtime arguments out of *params*.

        :param params: The flat parameter vector.
        :return: The gate as a :class:`~quax.Unitary`.
        """
        resolved = [
            params[argument.index] if isinstance(argument, Slot) else argument.value for argument in self.arguments
        ]
        gate = self.gate_fn(*resolved)
        if not isinstance(gate, Unitary):
            raise CircuitTypeError(
                f"Gate constructor {self.name} returned {type(gate).__name__}, not a Unitary.",
                kind=CircuitErrorKind.NON_UNITARY_OP,
            )
        return gate


#: Anything that may appear as an operation in a circuit: a built operator, or a gate that will
#: be built from parameters.  The general case, hence the unqualified name.
CircuitOp: TypeAlias = ConstantOp | ParameterizedGate

#: One operation: an operation together with the register indices it acts on, in operand order.
Placement: TypeAlias = tuple[CircuitOp, tuple[int, ...]]


@final
@dataclass(frozen=True, kw_only=True)
class Circuit:
    """An ordered sequence of operations placed on a qudit register.

    The general circuit type, and the one a front end produces: an operation is either an
    operator that is already built or a :class:`ParameterizedGate` still waiting for its
    arguments.  A circuit is structurally complete either way — every placement fixed, every
    dependency known — which is what lets a merge be planned and an operator stack be batched
    before any parameter has a value.

    :meth:`to_constant_circuit` builds every gate and returns a :class:`ConstantCircuit`, the
    narrower type that composing, merging and representation changes consume.

    **Every gate argument owns its own slot.** No two :class:`ParameterizedGate` arguments share one,
    which makes ``jax.grad`` unambiguous: entry ``k`` of the gradient is the derivative with
    respect to one specific gate argument at one specific point in the circuit, never an
    implicit sum over several. A front end whose source language *does* share a parameter
    across gates (a Quil memory reference used in twenty places, say) maps that one value onto
    twenty slots on the way in; differentiating through that map sums the contributions back
    up, exactly and without the front end doing arithmetic. See :attr:`param_owners`.

    The invariant is enforced rather than assumed, because nothing downstream would catch its
    violation: building gates from a shared slot works perfectly well, and the forward pass
    would stay correct while that slot's gradient quietly became a sum.

    :param dims: Per-qudit dimensions of the register, e.g. ``(2, 2, 3)``.
    :param ops: The operations, each an ``(operation, subsystem)`` pair, in application order.
    :param num_params: Length of the parameter vector :meth:`bind` expects.
    """

    dims: tuple[int, ...]
    ops: tuple[Placement, ...]
    num_params: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(int(d) for d in self.dims))
        object.__setattr__(self, "ops", tuple((op, tuple(int(q) for q in sub)) for op, sub in self.ops))
        object.__setattr__(self, "num_params", int(self.num_params))

        if any(d < 1 for d in self.dims):
            raise ValueError(f"Register dimensions must be positive, got {self.dims}.")
        if self.num_params < 0:
            raise ValueError(f"num_params must be non-negative, got {self.num_params}.")

        num_qudits = len(self.dims)
        for index, (op, subsystem) in enumerate(self.ops):
            validate_placement_indices(index, subsystem, num_qudits, self.dims)
            # A ParameterizedGate's dimensions are unknown until it is built, so the operator-vs-register
            # dimension check belongs to bind(), where ConstantCircuit performs it on the real operators.
            if isinstance(op, ParameterizedGate):
                continue
            if len(op.dims[1]) != len(subsystem):
                raise CircuitValueError(
                    f"Operation {index} acts on {len(op.dims[1])} qudit(s) but is placed on "
                    f"{len(subsystem)} register position(s) {subsystem}.",
                    kind=CircuitErrorKind.OPERATOR_ARITY_MISMATCH,
                    op_index=index,
                    subsystem=subsystem,
                )

        self._validate_parameter_slots()

    def _validate_parameter_slots(self) -> None:
        """Enforce that the slots are a permutation of ``range(num_params)``."""
        used = [slot for op, _ in self.ops if isinstance(op, ParameterizedGate) for slot in op.free_slots]
        if sorted(used) == list(range(self.num_params)):
            return
        duplicated = sorted({slot for slot in used if used.count(slot) > 1})
        missing = sorted(set(range(self.num_params)) - set(used))
        out_of_range = sorted({slot for slot in used if slot >= self.num_params})
        detail = []
        if duplicated:
            detail.append(f"slot(s) {duplicated} claimed by more than one gate argument")
        if missing:
            detail.append(f"slot(s) {missing} claimed by none")
        if out_of_range:
            detail.append(f"slot(s) {out_of_range} beyond num_params={self.num_params}")
        raise CircuitValueError(
            "Parameter slots must be a permutation of "
            f"0..{self.num_params - 1}, each owned by exactly one gate argument: "
            + "; ".join(detail or [f"got {len(used)} reference(s) for num_params={self.num_params}"])
            + ". Each ParameterizedGate argument owns an independent slot so that a gradient entry "
            "refers to one gate occurrence; map a shared source parameter onto several slots "
            "instead of reusing one.",
            kind=CircuitErrorKind.PARAM_SLOT_NOT_UNIQUE,
        )

    # ----- display -----

    def __str__(self) -> str:
        return f"Circuit(dims={self.dims}, num_ops={self.num_ops}, num_params={self.num_params})"

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self) -> Iterator[Placement]:
        return iter(self.ops)

    def __getitem__(self, index: int) -> Placement:
        """Return one operation as an ``(operation, subsystem)`` pair."""
        return self.ops[index]

    # ----- structure -----

    @property
    def num_ops(self) -> int:
        """The number of operations."""
        return len(self.ops)

    @property
    def num_qudits(self) -> int:
        """The number of qudits in the register."""
        return len(self.dims)

    @property
    def dim(self) -> int:
        """The total Hilbert-space dimension of the register."""
        return reduce(mul, self.dims, 1)

    @cached_property
    def subsystems(self) -> tuple[tuple[int, ...], ...]:
        """The register indices each operation acts on, in operand order."""
        return tuple(subsystem for _, subsystem in self.ops)

    @cached_property
    def operations(self) -> tuple[CircuitOp, ...]:
        """The operations, in application order.  Gates may be unbuilt."""
        return tuple(op for op, _ in self.ops)

    @cached_property
    def is_constant(self) -> bool:
        """Whether every operation is already built, so no parameters are needed."""
        return not any(isinstance(op, ParameterizedGate) for op, _ in self.ops)

    @cached_property
    def param_owners(self) -> tuple[tuple[int, int], ...]:
        """For each slot, the ``(operation index, argument index)`` that owns it.

        The inverse of the slot assignment, and what makes a gradient entry self-describing:
        ``param_owners[k]`` says which operation's which argument entry ``k`` differentiates.
        Well defined precisely because slots are not shared.
        """
        owners: list[tuple[int, int]] = [(-1, -1)] * self.num_params
        for op_index, (op, _) in enumerate(self.ops):
            if not isinstance(op, ParameterizedGate):
                continue
            for position, argument in enumerate(op.arguments):
                if isinstance(argument, Slot):
                    owners[argument.index] = (op_index, position)
        return tuple(owners)

    @cached_property
    def instrument_indices(self) -> tuple[int, ...]:
        """Indices of operations that are :class:`~quax.QuantumInstrument`."""
        return tuple(i for i, (op, _) in enumerate(self.ops) if isinstance(op, QuantumInstrument))

    # ----- construction -----

    @classmethod
    def from_ops(
        cls,
        ops: Sequence[Placement],
        num_params: int,
        num_qudits: int | None = None,
        default_dim: int = 2,
    ) -> "Circuit":
        """Build a circuit, inferring register dimensions from the *concrete* operators.

        A :class:`ParameterizedGate` contributes nothing to the inference — its dimensions are unknown
        until it is built — so a qudit touched only by gate calls takes ``default_dim``.  Pass
        ``dims`` to the constructor directly when that is not what you want.

        :param ops: The operations, in application order.
        :param num_params: Length of the parameter vector.
        :param num_qudits: Register size.  Defaults to one past the largest index used.
        :param default_dim: Dimension for a qudit no already-built operator determines.
        :return: The circuit.
        """
        if num_qudits is None:
            num_qudits = 1 + max((q for _, sub in ops for q in sub), default=-1)
        dims = [0] * num_qudits
        for op, subsystem in ops:
            if isinstance(op, ParameterizedGate):
                continue
            for qudit, d in zip(subsystem, op.dims[1]):  # not strict; see ConstantCircuit.infer_dims
                dims[qudit] = max(dims[qudit], d)
        return cls(dims=tuple(d or default_dim for d in dims), ops=tuple(ops), num_params=num_params)

    def with_ops(self, ops: Sequence[Placement], num_params: int | None = None) -> "Circuit":
        """Return a circuit with the same register and different operations.

        :param ops: The replacement operations.
        :param num_params: New parameter count; defaults to the current one.
        :return: The new circuit.
        """
        return Circuit(
            dims=self.dims,
            ops=tuple(ops),
            num_params=self.num_params if num_params is None else num_params,
        )

    # ----- binding -----

    def to_constant_circuit(self, params: Array | None = None) -> ConstantCircuit:
        """Build every gate and return a :class:`ConstantCircuit`.

        This is the only bridge between the two types, and the inverse of
        :meth:`ConstantCircuit.to_circuit`.  Everything that consumes already-built operators —
        merging, composing, representation changes — takes a ``ConstantCircuit``.

        :param params: The flat parameter vector.  May be omitted only when
            ``num_params == 0``.
        :return: The circuit, with every gate built.
        :raises CircuitValueError: If *params* has the wrong length, or is omitted for a
            circuit that takes parameters.
        """
        params = self.validate_params(params)
        return ConstantCircuit(
            dims=self.dims,
            ops=tuple(
                (op(params) if isinstance(op, ParameterizedGate) else op, subsystem) for op, subsystem in self.ops
            ),
        )

    def validate_params(self, params: Array | None) -> Array:
        """Return *params* as an array, checked against :attr:`num_params`.

        Reported here rather than left to the indexing that would otherwise fail: a gather out
        of range surfaces as an opaque XLA message far from the mistake, and a vector that is
        too long is silently ignored.

        :param params: The candidate vector, or ``None`` for a parameter-free circuit.
        :return: The validated vector.
        :raises CircuitValueError: If the length is wrong or the vector is missing.
        """
        if params is None:
            if self.num_params:
                raise CircuitValueError(
                    f"This circuit has {self.num_params} parameter(s); params cannot be omitted.",
                    kind=CircuitErrorKind.PARAM_COUNT_MISMATCH,
                )
            return jnp.zeros((0,), dtype=float)
        params = jnp.asarray(params)
        if params.shape != (self.num_params,):
            raise CircuitValueError(
                f"Expected {self.num_params} parameter(s) for this circuit, got shape {tuple(params.shape)}.",
                kind=CircuitErrorKind.PARAM_COUNT_MISMATCH,
            )
        return params

    # ----- representation changes -----

    def collapse_instruments(self) -> "Circuit":
        """Replace every :class:`~quax.QuantumInstrument` with its total channel.

        Density-matrix evolution describes the unconditioned dynamics, in which a measurement
        is a dephasing channel and the outcome labels are discarded.  Instruments also cannot
        be fused, so collapsing them *before* planning is what lets a measurement merge with
        its neighbours like any other superoperator.

        Only concrete operations are touched: a :class:`ParameterizedGate` always builds a unitary.

        :return: A circuit with no instruments.
        """
        if not self.instrument_indices:
            return self
        collapsed: list[Placement] = [
            (op.total_channel() if isinstance(op, QuantumInstrument) else op, subsystem) for op, subsystem in self.ops
        ]
        return self.with_ops(collapsed)

    # ----- combination -----

    def concat(self, other: "Circuit") -> "Circuit":
        """Append *other*'s operations to this circuit's, renumbering its parameter slots.

        The two circuits must share a register.  ``other``'s slots are shifted up by this
        circuit's :attr:`num_params` so that the result still owns one slot per gate argument;
        the combined vector is this circuit's parameters followed by ``other``'s.

        :param other: The circuit to append.
        :return: The concatenation.
        :raises ValueError: If the registers differ.
        """
        if self.dims != other.dims:
            raise ValueError(f"Cannot concatenate circuits over different registers: {self.dims} and {other.dims}.")
        offset = self.num_params
        shifted = [(shift_slots(op, offset), subsystem) for op, subsystem in other.ops]
        return Circuit(
            dims=self.dims,
            ops=self.ops + tuple(shifted),
            num_params=self.num_params + other.num_params,
        )

    def __add__(self, other: "Circuit") -> "Circuit":
        """``a + b`` is :meth:`concat`."""
        if not isinstance(other, Circuit):
            return NotImplemented
        return self.concat(other)


def shift_slots(op: CircuitOp, offset: int) -> CircuitOp:
    """Return *op* with every parameter slot shifted up by *offset*.

    :param op: The operation.  An already-built operator is returned unchanged.
    :param offset: How far to shift.
    :return: The shifted operation.
    """
    if not isinstance(op, ParameterizedGate) or offset == 0:
        return op
    return ParameterizedGate(
        gate_fn=op.gate_fn,
        arguments=tuple(
            Slot(index=argument.index + offset) if isinstance(argument, Slot) else argument for argument in op.arguments
        ),
    )


def concat(*circuits: Circuit) -> Circuit:
    """Concatenate circuits over a shared register, renumbering parameter slots.

    :param circuits: The circuits, in application order.  At least one is required.
    :return: The concatenation.
    :raises ValueError: If no circuits are given, or their registers differ.
    """
    if not circuits:
        raise ValueError("Cannot concatenate zero circuits; a register cannot be inferred.")
    return reduce(lambda a, b: a.concat(b), circuits)


def tile(op: ConstantOp, placements: Iterable[tuple[int, ...]]) -> tuple[ConstantPlacement, ...]:
    """Place one operator on several subsystems.

    A block that will be applied repeatedly can be built once as its own :class:`ConstantCircuit`,
    folded with :meth:`ConstantCircuit.compose`, and then placed wherever it is needed::

        >>> block = ConstantCircuit.from_ops([(gates.H, (0,)), (gates.CNOT, (0, 1))])
        >>> ops = tile(block.compose(), [(0, 1), (2, 3)])

    Note that pre-composing overrides the merge planner: a wide dense operator may be slower
    than letting :meth:`MergePlan.greedy` fuse the block's gates into narrower groups.  Reach
    for this when the block is known to be worth folding, not by default.

    :param op: The operator to place.
    :param placements: The subsystems to place it on, in application order.
    :return: The placements, ready to pass to a circuit constructor.
    """
    return tuple((op, tuple(int(q) for q in subsystem)) for subsystem in placements)


def random_circuit(
    dims: tuple[int, ...],
    num_ops: int,
    key: Array,
    *,
    gate_probability: float = 0.5,
    max_arity: int = 2,
) -> Circuit:
    """Generate a random circuit mixing single-argument gate calls with concrete unitaries.

    Every generated gate call is a single-qubit rotation, so a qudit of dimension greater than
    two is only ever touched by an already-built operator.

    :param dims: Per-qudit dimensions of the register.
    :param num_ops: The number of operations to generate.
    :param key: A JAX PRNG key.
    :param gate_probability: The probability that an operation is a parametric gate call
        rather than a concrete random unitary.
    :param max_arity: The largest number of qudits one concrete operation may act on.
    :return: The circuit.
    """
    from .gates import RX, RY, RZ

    rotations = (RX, RY, RZ)
    num_qudits = len(dims)
    if num_qudits == 0:
        raise ValueError("Cannot generate a circuit over an empty register.")
    qubit_positions = [q for q, d in enumerate(dims) if d == 2]
    max_arity = min(max_arity, num_qudits)

    ops: list[Placement] = []
    slot = 0
    for _ in range(num_ops):
        key, kind_key, which_key, subsystem_key, op_key = jax.random.split(key, 5)
        parametric = qubit_positions and float(jax.random.uniform(kind_key)) < gate_probability
        if parametric:
            gate_fn = rotations[int(jax.random.randint(which_key, (), 0, len(rotations)))]
            qudit = qubit_positions[int(jax.random.randint(subsystem_key, (), 0, len(qubit_positions)))]
            ops.append((ParameterizedGate(gate_fn=gate_fn, arguments=(Slot(index=slot),)), (qudit,)))
            slot += 1
        else:
            arity = int(jax.random.randint(which_key, (), 1, max_arity + 1))
            subsystem = tuple(int(q) for q in jax.random.choice(subsystem_key, num_qudits, (arity,), replace=False))
            op_dims = tuple(dims[q] for q in subsystem)
            ops.append((random_unitary((op_dims, op_dims), key=op_key), subsystem))
    return Circuit(dims=tuple(dims), ops=tuple(ops), num_params=slot)
