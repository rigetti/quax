#!/usr/bin/env python
"""Standalone benchmark: batch scaling of targeted_apply_kraus_map_trajectory.

Measures wall-clock time for 16Q and 20Q systems with a 2Q operator,
sweeping ensemble sizes to confirm whether batching improves per-state throughput.

Run:
    JAX_ENABLE_X64=1 poetry run python benchmarks/batch_scaling.py
"""

import time
from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp

import quax as qx

SEED = 42
N_WARMUP = 3
N_TRIALS = 10


def make_state_vector(dims, ensemble_size=()):
    key = jax.random.key(SEED)
    sv = qx.random_state_vector(dims=dims, key=key, size=ensemble_size)
    sv.data.block_until_ready()
    return sv


def make_depolarizing_kraus(gate_dims, truncate=False):
    s = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=gate_dims)
    km = qx.superop_to_kraus(s)
    if truncate:
        km = qx.truncate_kraus(km)
    km.data.block_until_ready()
    return km


def make_unitary(gate_dims):
    key = jax.random.key(SEED + 1)
    u = qx.random_unitary(dims=(gate_dims, gate_dims), key=key)
    u.data.block_until_ready()
    return u


def bench(fn, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    """Benchmark fn(), returning median time in seconds."""
    # Warmup (includes JIT compilation)
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]  # median


def run_kraus_trajectory_benchmark():
    print("=" * 80)
    print("BENCHMARK: targeted_apply_kraus_map_trajectory  batch scaling")
    print("=" * 80)

    # Use different ensemble sizes per system to avoid OOM at 20Q
    system_configs = [
        ("16Q", (2,) * 16, [(), (4,), (16,), (64,), (100,)]),
        ("20Q", (2,) * 20, [(), (4,), (16,)]),
    ]
    gate_dims = (2, 2)  # 2Q operator
    subsystem = (0, 1)

    for truncate in [False, True]:
        trunc_label = "truncated" if truncate else "full"
        kraus = make_depolarizing_kraus(gate_dims, truncate=truncate)
        n_kraus = kraus.data.shape[kraus.num_ensemble_dims]
        print(f"\n--- Kraus map: depolarizing 2Q, {trunc_label}, n_kraus={n_kraus} ---")
        print(f"{'System':<8} {'Ensemble':<12} {'Total (ms)':<14} {'Per-state (ms)':<16} {'Ratio vs 1':<12}")
        print("-" * 62)

        for sys_label, dims, ensemble_sizes in system_configs:
            base_time = None
            for ens in ensemble_sizes:
                n_states = reduce(mul, ens, 1)
                psi = make_state_vector(dims, ens)
                key = jax.random.key(SEED + 100)

                def fn():
                    r = qx.targeted_apply_kraus_map_trajectory(kraus, psi, key, subsystem)
                    r.data.block_until_ready()

                t = bench(fn)
                per_state = t / n_states * 1000  # ms
                total_ms = t * 1000

                if base_time is None:
                    base_time = per_state
                    ratio = 1.0
                else:
                    ratio = per_state / base_time

                ens_str = str(ens) if ens else "()"
                print(f"{sys_label:<8} {ens_str:<12} {total_ms:<14.2f} {per_state:<16.3f} {ratio:<12.2f}")


def run_unitary_benchmark():
    print("\n" + "=" * 80)
    print("BENCHMARK: targeted_apply_unitary  batch scaling (comparison)")
    print("=" * 80)

    system_configs = [
        ("16Q", (2,) * 16, [(), (4,), (16,), (64,), (100,)]),
        ("20Q", (2,) * 20, [(), (4,), (16,)]),
    ]
    gate_dims = (2, 2)
    subsystem = (0, 1)

    gate = make_unitary(gate_dims)
    print(f"\n--- Unitary 2Q gate ---")
    print(f"{'System':<8} {'Ensemble':<12} {'Total (ms)':<14} {'Per-state (ms)':<16} {'Ratio vs 1':<12}")
    print("-" * 62)

    for sys_label, dims, ensemble_sizes in system_configs:
        base_time = None
        for ens in ensemble_sizes:
            n_states = reduce(mul, ens, 1)
            psi = make_state_vector(dims, ens)

            def fn():
                r = qx.targeted_apply_unitary(gate, psi, subsystem)
                r.data.block_until_ready()

            t = bench(fn)
            per_state = t / n_states * 1000
            total_ms = t * 1000

            if base_time is None:
                base_time = per_state
                ratio = 1.0
            else:
                ratio = per_state / base_time

            ens_str = str(ens) if ens else "()"
            print(f"{sys_label:<8} {ens_str:<12} {total_ms:<14.2f} {per_state:<16.3f} {ratio:<12.2f}")


def run_rdm_kraus_trajectory_benchmark():
    print("\n" + "=" * 80)
    print("BENCHMARK: targeted_apply_kraus_map_trajectory_rdm  batch scaling")
    print("=" * 80)

    system_configs = [
        ("16Q", (2,) * 16, [(), (4,), (16,), (64,), (100,)]),
        ("20Q", (2,) * 20, [(), (4,), (16,), (64,)]),
    ]
    gate_dims = (2, 2)
    subsystem = (0, 1)

    for truncate in [False, True]:
        trunc_label = "truncated" if truncate else "full"
        kraus = make_depolarizing_kraus(gate_dims, truncate=truncate)
        n_kraus = kraus.data.shape[kraus.num_ensemble_dims]
        print(f"\n--- Kraus map (RDM): depolarizing 2Q, {trunc_label}, n_kraus={n_kraus} ---")
        print(f"{'System':<8} {'Ensemble':<12} {'Total (ms)':<14} {'Per-state (ms)':<16} {'Ratio vs 1':<12}")
        print("-" * 62)

        for sys_label, dims, ensemble_sizes in system_configs:
            base_time = None
            for ens in ensemble_sizes:
                n_states = reduce(mul, ens, 1)
                psi = make_state_vector(dims, ens)
                key = jax.random.key(SEED + 100)

                def fn():
                    r = qx.targeted_apply_kraus_map_trajectory_rdm(kraus, psi, key, subsystem)
                    r.data.block_until_ready()

                t = bench(fn)
                per_state = t / n_states * 1000
                total_ms = t * 1000

                if base_time is None:
                    base_time = per_state
                    ratio = 1.0
                else:
                    ratio = per_state / base_time

                ens_str = str(ens) if ens else "()"
                print(f"{sys_label:<8} {ens_str:<12} {total_ms:<14.2f} {per_state:<16.3f} {ratio:<12.2f}")


def run_single_trajectory_benchmark():
    print("\n" + "=" * 80)
    print("BENCHMARK: targeted_apply_kraus_map_single_trajectory  (single state)")
    print("=" * 80)

    systems = [
        ("16Q", (2,) * 16),
        ("20Q", (2,) * 20),
    ]
    gate_dims_list = [(2,), (2, 2)]
    noise_levels = [0.01, 0.05, 0.1]

    for gate_dims in gate_dims_list:
        subsystem = tuple(range(len(gate_dims)))
        op_label = f"{len(gate_dims)}Qop"
        for noise_p in noise_levels:
            s = qx.depolarizing_channel_superoperator(jnp.array(noise_p), dims=gate_dims)
            kraus = qx.superop_to_kraus(s)
            kraus_trunc = qx.truncate_kraus(kraus)
            n_kraus_full = kraus.data.shape[0]
            n_kraus_trunc = kraus_trunc.data.shape[0]

            is_ul, smq = qx.classify_kraus_operators(kraus_trunc)
            n_ul = int(jnp.sum(is_ul))

            print(f"\n--- {op_label}, p={noise_p}, full={n_kraus_full}, trunc={n_kraus_trunc}, unitary_like={n_ul} ---")
            print(f"{'System':<8} {'Method':<14} {'Time (ms)':<12} {'vs original':<12}")
            print("-" * 50)

            for sys_label, dims in systems:
                psi = make_state_vector(dims)
                key = jax.random.key(SEED + 100)

                # Original (single state)
                def fn_orig():
                    r = qx.targeted_apply_kraus_map_trajectory(kraus_trunc, psi, key, subsystem)
                    r.data.block_until_ready()
                t_orig = bench(fn_orig) * 1000

                # RDM (single state)
                def fn_rdm():
                    r = qx.targeted_apply_kraus_map_trajectory_rdm(kraus_trunc, psi, key, subsystem)
                    r.data.block_until_ready()
                t_rdm = bench(fn_rdm) * 1000

                # Single trajectory (with while_loop early termination)
                def fn_single():
                    r = qx.targeted_apply_kraus_map_single_trajectory(kraus_trunc, psi, key, subsystem)
                    r.data.block_until_ready()
                t_single = bench(fn_single) * 1000

                print(f"{sys_label:<8} {'original':<14} {t_orig:<12.3f} {'1.00x':<12}")
                print(f"{sys_label:<8} {'rdm':<14} {t_rdm:<12.3f} {f'{t_orig/t_rdm:.2f}x':<12}")
                print(f"{sys_label:<8} {'single':<14} {t_single:<12.3f} {f'{t_orig/t_single:.2f}x':<12}")
                print()


if __name__ == "__main__":
    run_kraus_trajectory_benchmark()
    run_rdm_kraus_trajectory_benchmark()
    run_single_trajectory_benchmark()
    run_unitary_benchmark()
