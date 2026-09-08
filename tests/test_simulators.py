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

"""Tests for the differentiable simulators.

A simulator is a fast path, not a definition.  What a circuit *means* is already fixed by
:meth:`~quax.Circuit.compose`: fold every operation into one operator and apply it to the
initial state.  So correctness here is almost entirely one property —

    simulate(circuit) == apply(compose(circuit), |0>)

— checked against that independent path rather than against stored numbers, over random
circuits and across every merge budget.  Everything the simulator does that ``compose`` does
not (planning merges, batching gate construction, padding a stack, dispatching a switch) is
an optimisation, and an optimisation is exactly the kind of thing that is correct on the case
you thought of and wrong on the case you did not.  Sweeping ``max_subsystem_size`` matters
particularly: it changes the merge structure completely while leaving the answer invariant.

The remaining tests cover what ``compose`` cannot speak to: that ``jit`` and ``grad`` work and
agree with the eager path, that preparation happens once and is not captured by a trace, and
that each simulator rejects what it cannot represent.
"""

import jax
import jax.numpy as jnp
import pytest

import quax as qx

# ---------- helpers ----------


def rx(slot: int) -> qx.GateCall:
    """A single-argument ``RX`` reading one parameter slot."""
    return qx.GateCall(gate_fn=qx.gates.RX, param_indices=(slot,), concrete_values=(None,))


def ry(slot: int) -> qx.GateCall:
    """A single-argument ``RY`` reading one parameter slot."""
    return qx.GateCall(gate_fn=qx.gates.RY, param_indices=(slot,), concrete_values=(None,))


def composed_state_vector(circuit: qx.ParametricCircuit, params=None) -> qx.StateVector:
    """The reference pure state: fold the circuit, apply it to ``|0>``."""
    bound = circuit.bind(params)
    return qx.apply_unitary_to_state_vector(bound.compose(), qx.zero_state_vector(dims=bound.dims))


def composed_density_matrix(circuit: qx.ParametricCircuit, params=None) -> qx.DensityMatrix:
    """The reference mixed state: fold the circuit's channel, apply it to ``|0><0|``."""
    bound = circuit.bind(params).to_superops()
    return qx.apply_superop_to_density_matrix(bound.compose(), qx.zero_state_matrix(dims=bound.dims))


def assert_close(actual, expected, atol: float = 1e-11) -> None:
    """Assert two quantum objects agree, comparing raw data exactly."""
    assert actual.dims == expected.dims
    difference = float(jnp.max(jnp.abs(actual.data - expected.data)))
    assert difference < atol, f"states differ by {difference:g}"


MERGE_BUDGETS = [0, 1, 2, 3]
"""Every merge budget worth sweeping: 0 disables merging, 3 exceeds the usual two-qubit case."""


# ---------- the defining property ----------


class TestAgreesWithCompose:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("max_subsystem_size", MERGE_BUDGETS)
    def test_state_vector_matches_composed_unitary(self, seed, max_subsystem_size):
        circuit = qx.random_parametric_circuit((2,) * 4, 14, jax.random.key(seed))
        params = jax.random.uniform(jax.random.key(100 + seed), (circuit.num_params,)) * 6.0
        simulator = qx.StateVectorSimulator(circuit=circuit, max_subsystem_size=max_subsystem_size)
        assert_close(simulator.compute(params), composed_state_vector(circuit, params))

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("max_subsystem_size", MERGE_BUDGETS)
    def test_density_matrix_matches_composed_channel(self, seed, max_subsystem_size):
        circuit = qx.random_parametric_circuit((2,) * 3, 12, jax.random.key(seed))
        params = jax.random.uniform(jax.random.key(200 + seed), (circuit.num_params,)) * 6.0
        simulator = qx.DensityMatrixSimulator(circuit=circuit, max_subsystem_size=max_subsystem_size)
        assert_close(simulator.compute(params), composed_density_matrix(circuit, params))

    @pytest.mark.parametrize("max_subsystem_size", MERGE_BUDGETS)
    def test_a_noisy_circuit_matches_its_composed_channel(self, max_subsystem_size):
        circuit = qx.ParametricCircuit.from_ops(
            [
                (rx(0), (0,)),
                (qx.gates.CNOT, (0, 1)),
                (qx.channels.depolarizing(0.3, dims=(2,)), (1,)),
                (ry(1), (1,)),
                (qx.channels.amplitude_damping(0.2, dims=(2,)), (0,)),
            ],
            num_params=2,
        )
        params = jnp.array([0.7, 1.3])
        simulator = qx.DensityMatrixSimulator(circuit=circuit, max_subsystem_size=max_subsystem_size)
        assert_close(simulator.compute(params), composed_density_matrix(circuit, params))

    @pytest.mark.parametrize("max_subsystem_size", MERGE_BUDGETS)
    def test_a_qutrit_register_matches(self, max_subsystem_size):
        circuit = qx.ParametricCircuit.from_ops(
            [(qx.gates.TH, (0,)), (qx.gates.X, (1,)), (qx.gates.TSWAP, (0, 2)), (qx.gates.TX, (2,))],
            num_params=0,
        )
        assert circuit.dims == (3, 2, 3)
        simulator = qx.StateVectorSimulator(circuit=circuit, max_subsystem_size=max_subsystem_size)
        assert_close(simulator.compute(), composed_state_vector(circuit))

    @pytest.mark.parametrize("max_subsystem_size", MERGE_BUDGETS)
    def test_a_reversed_multi_qubit_gate_matches(self, max_subsystem_size):
        # A singleton group keeps its own operand order, so the simulator must apply the
        # matrix to permuted qudits rather than to the sorted subsystem.  A three-qubit gate
        # can never merge under the default budget, so this is the case that catches it.
        circuit = qx.ParametricCircuit.from_ops(
            [(qx.gates.X, (2,)), (qx.gates.CCNOT, (2, 1, 0)), (qx.gates.CNOT, (1, 0))],
            num_params=0,
        )
        simulator = qx.StateVectorSimulator(circuit=circuit, max_subsystem_size=max_subsystem_size)
        assert_close(simulator.compute(), composed_state_vector(circuit))

    def test_the_result_does_not_depend_on_the_merge_budget(self):
        circuit = qx.random_parametric_circuit((2,) * 4, 20, jax.random.key(11))
        params = jax.random.uniform(jax.random.key(12), (circuit.num_params,)) * 6.0
        results = [
            qx.StateVectorSimulator(circuit=circuit, max_subsystem_size=budget).compute(params)
            for budget in MERGE_BUDGETS
        ]
        for other in results[1:]:
            assert_close(other, results[0])


class TestMeasurement:
    def test_an_instrument_acts_as_its_total_channel(self):
        circuit = qx.ParametricCircuit.from_ops(
            [(qx.gates.H, (0,)), (qx.gates.MEASURE(dim=2), (0,))],
            num_params=0,
        )
        rho = qx.DensityMatrixSimulator(circuit=circuit).compute()
        # H then dephasing measurement: a maximally mixed single qubit.
        assert jnp.allclose(rho.matrix, jnp.eye(2) / 2, atol=1e-12)

    def test_instruments_are_collapsed_before_planning(self):
        # Collapsing first is what lets a measurement merge with its neighbours; leaving the
        # instrument in place would force it to stand alone and could raise during apply.
        circuit = qx.ParametricCircuit.from_ops(
            [(rx(0), (0,)), (qx.gates.MEASURE(dim=2), (0,)), (qx.gates.CNOT, (0, 1))],
            num_params=1,
        )
        simulator = qx.DensityMatrixSimulator(circuit=circuit)
        assert simulator.prepared_circuit.instrument_indices == ()
        assert_close(simulator.compute(jnp.array([0.9])), composed_density_matrix(circuit, jnp.array([0.9])))

    def test_the_state_stays_a_valid_density_matrix(self):
        circuit = qx.ParametricCircuit.from_ops(
            [
                (qx.gates.H, (0,)),
                (qx.gates.CNOT, (0, 1)),
                (qx.gates.MEASURE(dim=2), (0,)),
                (qx.channels.depolarizing(0.4, dims=(2,)), (1,)),
            ],
            num_params=0,
        )
        rho = qx.DensityMatrixSimulator(circuit=circuit).compute()
        assert jnp.allclose(jnp.trace(rho.matrix), 1.0, atol=1e-12)
        assert jnp.allclose(rho.matrix, rho.matrix.conj().T, atol=1e-12)
        assert float(jnp.min(jnp.linalg.eigvalsh(rho.matrix))) > -1e-12


# ---------- transformations ----------


class TestJitAndGrad:
    @pytest.fixture
    def circuit(self):
        return qx.ParametricCircuit.from_ops(
            [(rx(0), (0,)), (qx.gates.CNOT, (0, 1)), (ry(1), (1,)), (rx(2), (0,))],
            num_params=3,
        )

    @pytest.fixture
    def params(self):
        return jnp.array([0.3, 1.1, 2.4])

    def test_jit_agrees_with_eager(self, circuit, params):
        simulator = qx.StateVectorSimulator(circuit=circuit)
        assert_close(jax.jit(simulator.compute)(params), simulator.compute(params), atol=1e-13)

    def test_jit_first_then_eager(self, circuit, params):
        # Preparation must not be captured by a trace.  If any derived quantity were built
        # lazily inside the first ``jit``, it would hold values belonging to a trace that has
        # ended, and the eager call after it would fail with a leaked tracer.
        simulator = qx.StateVectorSimulator(circuit=circuit)
        jitted = jax.jit(simulator.compute)(params)
        assert_close(simulator.compute(params), jitted, atol=1e-13)

    def test_density_matrix_jit_agrees_with_eager(self, circuit, params):
        simulator = qx.DensityMatrixSimulator(circuit=circuit)
        assert_close(jax.jit(simulator.compute)(params), simulator.compute(params), atol=1e-13)

    def test_gradient_matches_finite_differences(self, circuit, params):
        simulator = qx.StateVectorSimulator(circuit=circuit)

        def cost(values):
            return jnp.abs(simulator.compute(values).data[0, 0]) ** 2

        analytic = jax.grad(cost)(params)
        step = 1e-6
        numeric = jnp.array(
            [(cost(params.at[i].add(step)) - cost(params.at[i].add(-step))) / (2 * step) for i in range(params.size)]
        )
        assert jnp.allclose(analytic, numeric, atol=1e-6)

    def test_gradient_is_per_gate_occurrence(self, circuit, params):
        # Slots 0 and 2 both drive an RX on qudit 0. Because they are separate slots, the
        # gradient reports them separately rather than as one summed entry -- which is the
        # entire reason the slot invariant exists.
        simulator = qx.StateVectorSimulator(circuit=circuit)
        gradient = jax.grad(lambda v: jnp.abs(simulator.compute(v).data[0, 0]) ** 2)(params)
        assert gradient.shape == (3,)
        assert circuit.param_owners[0] == (0, 0)
        assert circuit.param_owners[2] == (3, 0)

    def test_a_shared_source_parameter_sums_by_the_chain_rule(self, circuit):
        # A front end whose source shares one value across gates maps it onto several slots.
        # Differentiating through that map recovers the summed derivative for free, so the
        # front end never performs the summation itself.
        simulator = qx.StateVectorSimulator(circuit=circuit)
        scatter = jnp.array([0, 1, 0])  # source value 0 drives slots 0 and 2.
        source = jnp.array([0.5, 1.2])

        def cost(values):
            return jnp.abs(simulator.compute(values).data[0, 0]) ** 2

        per_slot = jax.grad(cost)(source[scatter])
        per_source = jax.grad(lambda s: cost(s[scatter]))(source)
        assert jnp.allclose(per_source[0], per_slot[0] + per_slot[2], atol=1e-10)
        assert jnp.allclose(per_source[1], per_slot[1], atol=1e-10)

    def test_jit_of_grad_agrees_with_grad(self, circuit, params):
        simulator = qx.StateVectorSimulator(circuit=circuit)
        cost = lambda v: jnp.abs(simulator.compute(v).data[0, 0]) ** 2
        assert jnp.allclose(jax.jit(jax.grad(cost))(params), jax.grad(cost)(params), atol=1e-12)

    def test_density_matrix_gradients_are_finite(self, circuit, params):
        simulator = qx.DensityMatrixSimulator(circuit=circuit)
        gradient = jax.grad(lambda v: jnp.real(simulator.compute(v).matrix[0, 0]))(params)
        assert bool(jnp.all(jnp.isfinite(gradient)))

    def test_vmap_over_a_batch_of_parameters(self, circuit):
        simulator = qx.StateVectorSimulator(circuit=circuit)
        batch = jax.random.uniform(jax.random.key(5), (4, circuit.num_params))
        batched = jax.vmap(simulator.compute)(batch)
        for i in range(4):
            assert jnp.allclose(batched.data[i], simulator.compute(batch[i]).data, atol=1e-12)


# ---------- structure and preparation ----------


class TestPreparation:
    def test_the_stack_has_one_matrix_per_merge_group(self):
        circuit = qx.random_parametric_circuit((2,) * 3, 12, jax.random.key(4))
        params = jnp.linspace(0.0, 1.0, circuit.num_params)
        simulator = qx.StateVectorSimulator(circuit=circuit)
        stack = simulator.operator_stack(params)
        assert stack.shape == (simulator.plan.num_groups, simulator.d_max, simulator.d_max)

    def test_the_density_matrix_stack_is_squared_in_width(self):
        circuit = qx.ParametricCircuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))], num_params=0)
        simulator = qx.DensityMatrixSimulator(circuit=circuit)
        assert simulator.operator_stack().shape[-1] == simulator.d_max**2

    def test_a_parameter_free_circuit_materialises_its_stack_eagerly(self):
        circuit = qx.ParametricCircuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))], num_params=0)
        simulator = qx.StateVectorSimulator(circuit=circuit)
        assert simulator._constant_stack is not None

    def test_a_parametric_circuit_has_no_constant_stack(self):
        simulator = qx.StateVectorSimulator(circuit=qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=1))
        assert simulator._constant_stack is None

    def test_the_traced_graph_does_not_grow_with_circuit_depth(self):
        # The reason GateCall exists: same-kind gates are built under one vmap, and merged
        # operations dispatch through a switch, so neither the number of gates nor the number
        # of merge groups puts equations into the graph.
        def layered(depth):
            ops, slot = [], 0
            for layer in range(depth):
                for qubit in range(4):
                    ops.append((rx(slot), (qubit,)))
                    slot += 1
                for qubit in range(layer % 2, 3, 2):
                    ops.append((qx.gates.CZ, (qubit, qubit + 1)))
            return qx.ParametricCircuit.from_ops(ops, num_params=slot)

        counts = []
        for depth in (2, 8):
            circuit = layered(depth)
            simulator = qx.StateVectorSimulator(circuit=circuit)
            jaxpr = jax.make_jaxpr(simulator.compute)(jnp.zeros(circuit.num_params))
            counts.append(len(jaxpr.jaxpr.eqns))
        assert counts[0] == counts[1], f"graph grew with depth: {counts}"

    def test_the_merge_plan_covers_every_operation_once(self):
        circuit = qx.random_parametric_circuit((2,) * 4, 18, jax.random.key(6))
        plan = qx.StateVectorSimulator(circuit=circuit).plan
        assert sorted(plan.flat_order) == list(range(circuit.num_ops))
        assert plan.group_start[-1] == circuit.num_ops
        assert len(plan.group_start) == plan.num_groups + 1

    def test_simulators_are_hashable_despite_holding_arrays(self):
        # ``eq=False`` keeps identity hashing; a synthesised ``__hash__`` over JAX array fields
        # would raise, and would do so far from the cause.
        simulator = qx.StateVectorSimulator(circuit=qx.ParametricCircuit.from_ops([(qx.gates.H, (0,))], num_params=0))
        assert {simulator: "usable as a key"}[simulator] == "usable as a key"


class TestEmptyAndTrivial:
    def test_an_empty_circuit_returns_the_initial_state_vector(self):
        circuit = qx.ParametricCircuit(dims=(2, 2), ops=(), num_params=0)
        simulator = qx.StateVectorSimulator(circuit=circuit)
        assert_close(simulator.compute(), qx.zero_state_vector(dims=(2, 2)))

    def test_an_empty_circuit_returns_the_initial_density_matrix(self):
        circuit = qx.ParametricCircuit(dims=(2,), ops=(), num_params=0)
        assert_close(qx.DensityMatrixSimulator(circuit=circuit).compute(), qx.zero_state_matrix(dims=(2,)))

    def test_a_single_gate_circuit(self):
        circuit = qx.ParametricCircuit.from_ops([(qx.gates.X, (0,))], num_params=0)
        assert jnp.allclose(qx.StateVectorSimulator(circuit=circuit).compute().matrix, jnp.array([0.0, 1.0]))

    def test_an_idle_qudit_is_left_alone(self):
        circuit = qx.ParametricCircuit(dims=(2, 2), ops=((qx.gates.X, (0,)),), num_params=0)
        state = qx.StateVectorSimulator(circuit=circuit).compute()
        assert jnp.allclose(state.matrix, jnp.array([0.0, 0.0, 1.0, 0.0]))


class TestUnitaryReadout:
    def test_matches_the_composed_circuit(self):
        circuit = qx.random_parametric_circuit((2,) * 3, 10, jax.random.key(8))
        params = jax.random.uniform(jax.random.key(9), (circuit.num_params,)) * 6.0
        simulator = qx.StateVectorSimulator(circuit=circuit)
        unitary = simulator.unitary(params)
        difference = float(jnp.max(jnp.abs(unitary.matrix - circuit.bind(params).compose().matrix)))
        assert difference < 1e-11

    def test_applying_it_reproduces_compute(self):
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,)), (qx.gates.CNOT, (0, 1))], num_params=1)
        params = jnp.array([1.1])
        simulator = qx.StateVectorSimulator(circuit=circuit)
        applied = qx.apply_unitary_to_state_vector(simulator.unitary(params), qx.zero_state_vector(dims=circuit.dims))
        assert_close(applied, simulator.compute(params))

    def test_an_empty_circuit_gives_the_identity(self):
        circuit = qx.ParametricCircuit(dims=(2, 2), ops=(), num_params=0)
        unitary = qx.StateVectorSimulator(circuit=circuit).unitary()
        assert jnp.allclose(unitary.matrix, jnp.eye(4), atol=1e-12)


# ---------- rejection ----------


class TestValidation:
    @pytest.mark.parametrize(
        "operation",
        [
            qx.channels.depolarizing(0.2, dims=(2,)),
            qx.gates.MEASURE(dim=2),
            qx.gates.RESET(dim=2),
        ],
        ids=["channel", "instrument", "reset"],
    )
    def test_the_state_vector_simulator_rejects_non_unitary_operations(self, operation):
        circuit = qx.ParametricCircuit.from_ops([(qx.gates.H, (0,)), (operation, (0,))], num_params=0)
        with pytest.raises(qx.CircuitError, match="unitary operations only") as caught:
            qx.StateVectorSimulator(circuit=circuit)
        assert caught.value.kind is qx.CircuitErrorKind.NON_UNITARY_OP
        assert caught.value.op_index == 1
        assert caught.value.subsystem == (0,)

    def test_the_rejection_is_also_a_plain_type_error(self):
        # Front ends that predate the structured errors must keep working.
        circuit = qx.ParametricCircuit.from_ops([(qx.gates.RESET(dim=2), (0,))], num_params=0)
        with pytest.raises(TypeError):
            qx.StateVectorSimulator(circuit=circuit)

    def test_the_density_matrix_simulator_accepts_everything(self):
        circuit = qx.ParametricCircuit.from_ops(
            [
                (qx.gates.RESET(dim=2), (0,)),
                (qx.gates.MEASURE(dim=2), (0,)),
                (qx.channels.depolarizing(0.1, dims=(2,)), (0,)),
            ],
            num_params=0,
        )
        assert qx.DensityMatrixSimulator(circuit=circuit).compute() is not None

    def test_rejects_a_negative_merge_budget(self):
        circuit = qx.ParametricCircuit.from_ops([(qx.gates.H, (0,))], num_params=0)
        with pytest.raises(ValueError, match="max_subsystem_size"):
            qx.StateVectorSimulator(circuit=circuit, max_subsystem_size=-1)

    def test_compute_validates_the_parameter_vector(self):
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=1)
        simulator = qx.StateVectorSimulator(circuit=circuit)
        with pytest.raises(qx.CircuitError, match="Expected 1 parameter"):
            simulator.compute(jnp.zeros(2))

    def test_an_empty_circuit_still_validates_parameters(self):
        circuit = qx.ParametricCircuit(dims=(2,), ops=(), num_params=0)
        with pytest.raises(qx.CircuitError, match="Expected 0 parameter"):
            qx.StateVectorSimulator(circuit=circuit).compute(jnp.zeros(3))


class TestSimulateConvenience:
    def test_chooses_a_state_vector_for_a_unitary_circuit(self):
        circuit = qx.Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))])
        assert isinstance(qx.simulate(circuit), qx.StateVector)

    def test_chooses_a_density_matrix_when_a_channel_is_present(self):
        circuit = qx.Circuit.from_ops([(qx.gates.H, (0,)), (qx.channels.depolarizing(0.1, dims=(2,)), (0,))])
        assert isinstance(qx.simulate(circuit), qx.DensityMatrix)

    def test_accepts_a_parametric_circuit(self):
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=1)
        assert_close(qx.simulate(circuit, jnp.array([0.8])), composed_state_vector(circuit, jnp.array([0.8])))

    def test_agrees_with_constructing_the_simulator_directly(self):
        circuit = qx.Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))])
        lifted = qx.ParametricCircuit.from_circuit(circuit)
        assert_close(qx.simulate(circuit), qx.StateVectorSimulator(circuit=lifted).compute())


class TestCallable:
    def test_calling_a_simulator_computes(self):
        circuit = qx.ParametricCircuit.from_ops([(rx(0), (0,))], num_params=1)
        simulator = qx.StateVectorSimulator(circuit=circuit)
        params = jnp.array([0.6])
        assert_close(simulator(params), simulator.compute(params))
