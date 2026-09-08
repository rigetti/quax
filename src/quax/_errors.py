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

"""Structured errors for circuit construction and simulation.

Quax validates circuits; the *source language* that produced them lives somewhere else.  A
front end — a Quil compiler, say — therefore needs to catch a small number of quax failures
and re-raise them naming the instruction the user actually wrote.  Matching on message text
would make that contract unwritable, so the failures a front end is expected to intercept
carry a :class:`CircuitErrorKind` discriminant and, where quax knows it, the index of the
offending operation.

    >>> try:
    ...     simulator.compute()
    ... except CircuitError as error:
    ...     if error.kind is CircuitErrorKind.NON_UNITARY_OP:
    ...         raise MyFrontEndError(describe(error.op_index)) from error

The operation index is usually the more valuable half.  Quax can say *what* is wrong; only
the front end, which owns the mapping from its own source to circuit positions, can say
*which* instruction is wrong — and it can only do so if the error names a position.

Two concrete classes rather than one: quax raises ``ValueError`` at most of these sites and
``TypeError`` at others, a single class cannot be both, and collapsing them to one would
silently break existing ``except ValueError`` handlers.  So :class:`CircuitError` is the
common base — itself an ``Exception``, since Python will not catch a class that is not — and
each concrete class mixes it with the builtin that failure has always raised.  Catching
``CircuitError`` catches both; ``except ValueError`` and ``except TypeError`` keep working
exactly as before.

Kinds are added when a caller demonstrably needs to branch on one, not enumerated in
advance: an unused discriminant is a contract nobody is keeping.
"""

from abc import ABC
from enum import Enum, unique
from typing import final


@final
@unique
class CircuitErrorKind(Enum):
    """What went wrong, for callers that need to branch on it.

    Members are stable identifiers: a front end may switch on them, so renaming one is a
    breaking change.
    """

    SUBSYSTEM_OUT_OF_RANGE = "subsystem_out_of_range"
    """An operation names a qudit outside the register."""

    DUPLICATE_QUDIT = "duplicate_qudit"
    """An operation names the same qudit more than once."""

    OPERATOR_ARITY_MISMATCH = "operator_arity_mismatch"
    """An operator's qudit count does not match the subsystem it is placed on."""

    OPERATOR_EXCEEDS_REGISTER = "operator_exceeds_register"
    """An operator's dimension is larger than the register slot it is placed in."""

    NON_UNITARY_OP = "non_unitary_op"
    """A unitary-only consumer met an operation that is not a unitary."""

    INSTRUMENT_IN_MERGE_GROUP = "instrument_in_merge_group"
    """A merge group contains a quantum instrument, which cannot be fused."""

    PARAM_SLOT_NOT_UNIQUE = "param_slot_not_unique"
    """Parameter slots are not a permutation: a slot is shared, missing, or out of range."""

    PARAM_COUNT_MISMATCH = "param_count_mismatch"
    """A bound parameter vector has the wrong length for the circuit."""


class CircuitError(Exception, ABC):
    """Common base for circuit errors, carrying the structured payload.

    Catch this to catch every circuit error regardless of which builtin it also is.  It
    derives from :class:`Exception` because Python refuses to catch a class that does not —
    a payload-only mixin would be uncatchable, which would defeat the entire point.

    Abstract because it is not a thing to raise: the concrete classes below pin down which
    builtin a given failure also is, and that choice is part of each error's contract.
    """

    kind: CircuitErrorKind
    """Which failure this is."""

    op_index: int | None
    """Index of the offending operation in the circuit, when quax knows it."""

    subsystem: tuple[int, ...] | None
    """Register indices the offending operation acts on, when quax knows them."""

    @final
    def __init__(
        self,
        message: str,
        *,
        kind: CircuitErrorKind,
        op_index: int | None = None,
        subsystem: tuple[int, ...] | None = None,
    ) -> None:
        """Build the error.

        :param message: The human-readable message, complete on its own — a caller that does
            not know about :class:`CircuitErrorKind` must still get a useful error.
        :param kind: The discriminant.
        :param op_index: Index of the offending operation, if applicable.
        :param subsystem: Register indices of the offending operation, if applicable.
        """
        super().__init__(message)
        self.kind = kind
        self.op_index = op_index
        self.subsystem = subsystem


@final
class CircuitValueError(CircuitError, ValueError):
    """A circuit validation failure that is a bad *value*."""


@final
class CircuitTypeError(CircuitError, TypeError):
    """A circuit validation failure that is a bad *type*."""
