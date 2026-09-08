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

"""Tests for gate calls and parametric circuits.

The contract has three parts, and they fail in different ways:

* **Binding** must be faithful: ``circuit.bind(params)`` has to produce exactly the circuit a
  caller would have written by hand with those numbers substituted.  Every other test in the
  package assumes this, so it is checked against independently built gates rather than
  against another code path here.
* **Slot ownership** must be a bijection.  Each gate argument owns one parameter slot, which
  is what makes a gradient entry refer to a single gate occurrence.  Nothing downstream would
  notice a violation — building gates from a shared slot works fine and the forward pass stays
  correct — so the invariant is only ever enforced at construction, and these tests are the
  only thing standing behind it.
* **Batch keys** must group exactly the calls that trace to the same graph, and no others.
  Over-grouping is a correctness bug; under-grouping silently loses the compile-time win the
  whole type exists for.
"""

import jax
import jax.numpy as jnp
import pytest

import quax as qx

# ---------- helpers ----------


def rx(slot: int) -> qx.GateCall:
    """A single-argument ``RX`` reading one parameter slot."""
    return qx.GateCall(gate_fn=qx.gates.RX, param_indices=(slot,), concrete_values=(None,))


def constant_rx(angle: float) -> qx.GateCall:
    """An ``RX`` whose angle is pinned at build time."""
    return qx.GateCall(gate_fn=qx.gates.RX, param_indices=(None,), concrete_values=(angle,))


def assert_same_operator(actual, expected, atol: float = 1e-12) -> None:
    """Assert two operators have the same dimensions and matrix."""
    assert actual.dims == expected.dims
    difference = float(jnp.max(jnp.abs(actual.matrix - expected.matrix)))
    assert difference < atol, f"operators differ by {difference:g}"


# ---------- gate calls ----------


class TestGateCallValidation:
    def test_rejects_mismatched_argument_lengths(self):
        with pytest.raises(ValueError, match="same arguments"):
            qx.GateCall(gate_fn=qx.gates.RX, param_indices=(0, 1), concrete_values=(None,))

    def test_rejects_an_argument_that_is_both_slot_and_constant(self):
        with pytest.raises(ValueError, match="both a slot and a constant"):
            qx.GateCall(gate_fn=qx.gates.RX, param_indices=(0,), concrete_values=(1.0,))

    def test_rejects_an_argument_that_is_neither(self):
        with pytest.raises(ValueError, match="neither a slot nor a constant"):
            qx.GateCall(gate_fn=qx.gates.RX, param_indices=(None,), concrete_values=(None,))

    def test_rejects_a_negative_slot(self):
        with pytest.raises(ValueError, match="negative parameter slot"):
            qx.GateCall(gate_fn=qx.gates.RX, param_indices=(-1,), concrete_values=(None,))


class TestGateCallBuilding:
    def test_builds_the_same_gate_as_calling_the_constructor(self):
        params = jnp.array([0.37])
        assert_same_operator(rx(0)(params), qx.gates.RX(0.37))

    def test_reads_the_slot_it_was_given(self):
        params = jnp.array([0.1, 0.2, 0.3])
        assert_same_operator(rx(2)(params), qx.gates.RX(0.3))

    def test_a_constant_argument_ignores_the_parameter_vector(self):
        call = constant_rx(0.5)
        assert_same_operator(call(jnp.array([9.9])), qx.gates.RX(0.5))
        assert call.free_slots == ()

    def test_mixes_constant_and_runtime_arguments(self):
        # PHASEDRX(phase, angle): pin the phase, take the angle at runtime.
        call = qx.GateCall(gate_fn=qx.gates.PHASEDRX, param_indices=(None, 0), concrete_values=(0.25, None))
        assert call.free_slots == (0,)
        assert_same_operator(call(jnp.array([1.4])), qx.gates.PHASEDRX(0.25, 1.4))

    def test_building_is_differentiable(self):
        grad = jax.grad(lambda p: jnp.real(rx(0)(p).matrix[0, 0]))(jnp.array([0.6]))
        expected = -0.5 * jnp.sin(0.6 / 2)
        assert jnp.allclose(grad[0], expected, atol=1e-10)


class TestBatchKey:
    def test_calls_differing_only_in_slot_share_a_key(self):
        assert rx(0).batch_key == rx(7).batch_key

    def test_different_constructors_do_not_share_a_key(self):
        ry = qx.GateCall(gate_fn=qx.gates.RY, param_indices=(0,), concrete_values=(None,))
        assert rx(0).batch_key != ry.batch_key

    def test_a_pinned_constant_changes_the_key(self):
        assert constant_rx(0.5).batch_key != constant_rx(0.9).batch_key
        assert constant_rx(0.5).batch_key != rx(0).batch_key

    def test_which_argument_is_pinned_changes_the_key(self):
        pinned_first = qx.GateCall(gate_fn=qx.gates.PHASEDRX, param_indices=(None, 0), concrete_values=(0.25, None))
        pinned_second = qx.GateCall(gate_fn=qx.gates.PHASEDRX, param_indices=(0, None), concrete_values=(None, 0.25))
        assert pinned_first.batch_key != pinned_second.batch_key

    def test_the_key_is_hashable_and_survives_garbage_collection(self):
        # Keyed on the function object, not ``id``, which is reusable once an object dies.
        keys = {rx(i).batch_key for i in range(50)}
        assert len(keys) == 1


# ---------- slot ownership ----------


class TestParameterSlotInvariant:
    def test_accepts_one_slot_per_argument(self):
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,)), (rx(1), (1,))], num_params=2)
        assert circuit.num_params == 2

    def test_rejects_a_shared_slot(self):
        with pytest.raises(qx.CircuitError, match="more than one gate argument") as caught:
            qx.ParametricCircuit.from_ops([(rx(0), (0,)), (rx(0), (1,))], num_params=1)
        assert caught.value.kind is qx.CircuitErrorKind.PARAM_SLOT_NOT_UNIQUE

    def test_rejects_an_unclaimed_slot(self):
        with pytest.raises(qx.CircuitError, match="claimed by none"):
            qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=2)

    def test_rejects_a_slot_beyond_the_parameter_count(self):
        with pytest.raises(qx.CircuitError, match="beyond num_params"):
            qx.ParametricCircuit.from_ops([(rx(3), (0,))], num_params=1)

    def test_a_circuit_of_constants_needs_no_slots(self):
        circuit = qx.ParametricCircuit.from_ops([(constant_rx(0.5), (0,)), (qx.gates.H, (1,))], num_params=0)
        assert circuit.num_params == 0
        assert circuit.param_owners == ()

    def test_param_owners_inverts_the_slot_assignment(self):
        two_argument = qx.GateCall(gate_fn=qx.gates.PHASEDRX, param_indices=(2, 0), concrete_values=(None, None))
        circuit = qx.ParametricCircuit.from_ops(
            [(rx(1), (0,)), (two_argument, (1,))],
            num_params=3,
        )
        # slot 0 -> op 1 argument 1; slot 1 -> op 0 argument 0; slot 2 -> op 1 argument 0.
        assert circuit.param_owners == ((1, 1), (0, 0), (1, 0))

    def test_every_slot_has_exactly_one_owner(self):
        circuit = qx.random_parametric_circuit((2,) * 4, 25, jax.random.key(3))
        assert len(circuit.param_owners) == circuit.num_params
        assert len(set(circuit.param_owners)) == circuit.num_params


# ---------- binding ----------


class TestBind:
    def test_produces_the_hand_written_circuit(self):
        circuit = qx.ParametricCircuit.from_ops(
            [(rx(0), (0,)), (qx.gates.CNOT, (0, 1)), (rx(1), (1,))],
            num_params=2,
        )
        bound = circuit.bind(jnp.array([0.3, 0.8]))
        expected = qx.Circuit.from_ops([(qx.gates.RX(0.3), (0,)), (qx.gates.CNOT, (0, 1)), (qx.gates.RX(0.8), (1,))])
        assert bound.dims == expected.dims
        assert bound.subsystems == expected.subsystems
        for actual, wanted in zip(bound.operators, expected.operators, strict=True):
            assert_same_operator(actual, wanted)

    def test_preserves_operand_order(self):
        circuit = qx.ParametricCircuit.from_ops([(qx.gates.CNOT, (1, 0))], num_params=0)
        assert circuit.bind().subsystems == ((1, 0),)

    def test_concrete_operations_pass_through_untouched(self):
        channel = qx.channels.depolarizing(0.2, dims=(2,))
        circuit = qx.ParametricCircuit.from_ops([(channel, (0,))], num_params=0)
        assert circuit.bind().operators[0] is channel

    def test_omitting_parameters_is_allowed_only_when_there_are_none(self):
        qx.ParametricCircuit.from_ops([(qx.gates.H, (0,))], num_params=0).bind()
        with pytest.raises(qx.CircuitError, match="cannot be omitted") as caught:
            qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=1).bind()
        assert caught.value.kind is qx.CircuitErrorKind.PARAM_COUNT_MISMATCH

    @pytest.mark.parametrize("supplied", [0, 1, 3])
    def test_rejects_a_wrongly_sized_vector(self, supplied):
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,)), (rx(1), (1,))], num_params=2)
        with pytest.raises(qx.CircuitError, match="Expected 2 parameter"):
            circuit.bind(jnp.zeros(supplied))

    def test_binding_validates_dimensions_deferred_from_construction(self):
        # A gate call's dimensions are unknown until it is built, so a qutrit gate placed in a
        # qubit-sized slot survives construction -- but bind() must still catch it.
        qutrit_rotation = qx.GateCall(gate_fn=qx.gates.TRX01, param_indices=(0,), concrete_values=(None,))
        circuit = qx.ParametricCircuit(dims=(2,), ops=((qutrit_rotation, (0,)),), num_params=1)
        with pytest.raises(qx.CircuitError, match="does not fit the register") as caught:
            circuit.bind(jnp.array([0.5]))
        assert caught.value.kind is qx.CircuitErrorKind.OPERATOR_EXCEEDS_REGISTER

    def test_bind_is_jittable(self):
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=1)
        composed = jax.jit(lambda p: circuit.bind(p).compose().matrix)(jnp.array([0.4]))
        assert jnp.allclose(composed, qx.gates.RX(0.4).matrix, atol=1e-12)


class TestStructure:
    def test_reports_its_shape(self):
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,)), (qx.gates.CNOT, (0, 1))], num_params=1)
        assert circuit.num_ops == 2
        assert circuit.num_qudits == 2
        assert circuit.dim == 4
        assert circuit.subsystems == ((0,), (0, 1))
        assert not circuit.is_concrete

    def test_a_circuit_without_gate_calls_is_concrete(self):
        assert qx.ParametricCircuit.from_ops([(qx.gates.H, (0,))], num_params=0).is_concrete

    def test_from_circuit_round_trips(self):
        circuit = qx.random_circuit((2, 2, 2), 8, jax.random.key(1))
        lifted = qx.ParametricCircuit.from_circuit(circuit)
        assert lifted.num_params == 0
        assert lifted.dims == circuit.dims
        assert lifted.bind().subsystems == circuit.subsystems

    def test_dimension_inference_ignores_gate_calls(self):
        # A gate call cannot widen a slot, because its dimensions are not yet knowable.
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,)), (qx.gates.TH, (1,))], num_params=1)
        assert circuit.dims == (2, 3)

    def test_is_iterable_and_indexable(self):
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,)), (qx.gates.H, (1,))], num_params=1)
        assert len(circuit) == 2
        assert circuit[1][1] == (1,)
        assert [subsystem for _, subsystem in circuit] == [(0,), (1,)]

    def test_rejects_an_out_of_range_qudit(self):
        with pytest.raises(qx.CircuitError) as caught:
            qx.ParametricCircuit(dims=(2, 2), ops=((rx(0), (5,)),), num_params=1)
        assert caught.value.kind is qx.CircuitErrorKind.SUBSYSTEM_OUT_OF_RANGE
        assert caught.value.op_index == 0

    def test_rejects_a_repeated_qudit(self):
        with pytest.raises(qx.CircuitError) as caught:
            qx.ParametricCircuit(dims=(2, 2), ops=((qx.gates.CNOT, (1, 1)),), num_params=0)
        assert caught.value.kind is qx.CircuitErrorKind.DUPLICATE_QUDIT


class TestCollapseInstruments:
    def test_replaces_instruments_with_their_total_channel(self):
        circuit = qx.ParametricCircuit.from_ops(
            [(rx(0), (0,)), (qx.gates.MEASURE(dim=2), (1,))],
            num_params=1,
        )
        assert circuit.instrument_indices == (1,)
        collapsed = circuit.collapse_instruments()
        assert collapsed.instrument_indices == ()
        assert_same_operator(collapsed.ops[1][0], qx.gates.MEASURE(dim=2).total_channel())

    def test_leaves_gate_calls_alone(self):
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=1)
        assert circuit.collapse_instruments() is circuit

    def test_preserves_the_parameter_layout(self):
        circuit = qx.ParametricCircuit.from_ops(
            [(rx(0), (0,)), (qx.gates.MEASURE(dim=2), (1,)), (rx(1), (1,))],
            num_params=2,
        )
        collapsed = circuit.collapse_instruments()
        assert collapsed.num_params == 2
        assert collapsed.param_owners == circuit.param_owners


# ---------- combination ----------


class TestConcat:
    def test_appends_operations_in_order(self):
        first = qx.ParametricCircuit.from_ops([(qx.gates.H, (0,))], num_params=0, num_qudits=2)
        second = qx.ParametricCircuit.from_ops([(qx.gates.X, (1,))], num_params=0, num_qudits=2)
        joined = first.concat(second)
        assert joined.subsystems == ((0,), (1,))

    def test_renumbers_the_second_circuits_slots(self):
        block = qx.ParametricCircuit.from_ops([(rx(0), (0,)), (rx(1), (1,))], num_params=2)
        joined = block.concat(block)
        assert joined.num_params == 4
        # The invariant survives, which is the whole point of renumbering.
        slots = sorted(slot for op, _ in joined.ops if isinstance(op, qx.GateCall) for slot in op.free_slots)
        assert slots == [0, 1, 2, 3]

    def test_the_combined_vector_is_the_two_vectors_end_to_end(self):
        block = qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=1)
        joined = block.concat(block)
        bound = joined.bind(jnp.array([0.3, 1.2]))
        assert_same_operator(bound.operators[0], qx.gates.RX(0.3))
        assert_same_operator(bound.operators[1], qx.gates.RX(1.2))

    def test_concatenation_equals_writing_the_operations_out(self):
        left = qx.ParametricCircuit.from_ops([(rx(0), (0,)), (qx.gates.CNOT, (0, 1))], num_params=1)
        right = qx.ParametricCircuit.from_ops([(rx(0), (1,))], num_params=1)
        params = jnp.array([0.4, 1.1])
        joined = left.concat(right).bind(params).compose()
        written = qx.Circuit.from_ops(
            [(qx.gates.RX(0.4), (0,)), (qx.gates.CNOT, (0, 1)), (qx.gates.RX(1.1), (1,))]
        ).compose()
        assert_same_operator(joined, written)

    def test_the_plus_operator_concatenates(self):
        block = qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=1)
        assert (block + block).num_params == 2

    def test_rejects_mismatched_registers(self):
        two = qx.ParametricCircuit.from_ops([(qx.gates.H, (0,))], num_params=0, num_qudits=2)
        three = qx.ParametricCircuit.from_ops([(qx.gates.H, (0,))], num_params=0, num_qudits=3)
        with pytest.raises(ValueError, match="different registers"):
            two.concat(three)

    def test_the_free_function_folds_several_circuits(self):
        block = qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=1)
        assert qx.concat(block, block, block).num_params == 3

    def test_the_free_function_rejects_an_empty_argument_list(self):
        with pytest.raises(ValueError, match="zero circuits"):
            qx.concat()


class TestTile:
    def test_places_one_operator_on_several_subsystems(self):
        block = qx.Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))])
        placements = qx.tile(block.compose(), [(0, 1), (2, 3)])
        assert [subsystem for _, subsystem in placements] == [(0, 1), (2, 3)]
        assert placements[0][0] is placements[1][0]

    def test_a_tiled_block_equals_the_block_written_out_twice(self):
        block_ops = [(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))]
        folded = qx.Circuit.from_ops(list(qx.tile(qx.Circuit.from_ops(block_ops).compose(), [(0, 1), (2, 3)])))
        expanded = qx.Circuit.from_ops(block_ops + [(qx.gates.H, (2,)), (qx.gates.CNOT, (2, 3))])
        difference = float(jnp.max(jnp.abs(folded.compose().matrix - expanded.compose().matrix)))
        assert difference < 1e-12


class TestRandomParametricCircuit:
    @pytest.mark.parametrize("seed", range(4))
    def test_generates_a_valid_circuit(self, seed):
        circuit = qx.random_parametric_circuit((2, 2, 3), 15, jax.random.key(seed))
        assert circuit.num_ops == 15
        assert circuit.dims == (2, 2, 3)
        # Construction already enforced the slot invariant; confirm it is actually exercised.
        assert sorted(slot for op, _ in circuit.ops if isinstance(op, qx.GateCall) for slot in op.free_slots) == list(
            range(circuit.num_params)
        )

    def test_binds_without_error(self):
        circuit = qx.random_parametric_circuit((2, 2), 10, jax.random.key(7))
        bound = circuit.bind(jnp.linspace(0.0, 1.0, circuit.num_params))
        assert bound.num_ops == circuit.num_ops

    def test_gate_calls_never_land_on_a_qutrit(self):
        circuit = qx.random_parametric_circuit((3, 3), 12, jax.random.key(2))
        assert circuit.num_params == 0
