#!/usr/bin/env python
"""Quick benchmark: single-trajectory comparison only.

Run:
    JAX_ENABLE_X64=1 poetry run python benchmarks/bench_single.py
"""

import time

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


def bench(fn, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def main():
    print("=" * 70)
    print("SINGLE-STATE COMPARISON: original vs rdm vs single_trajectory")
    print("=" * 70)

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
            p_ul = float(jnp.sum(smq[is_ul]))

            print(f"\n--- {op_label}, p_error={noise_p}, "
                  f"n_kraus={n_kraus_full}->{n_kraus_trunc}, "
                  f"unitary_like={n_ul} (p_cum={p_ul:.4f}) ---")
            print(f"{'System':<8} {'Method':<14} {'Time (ms)':<12} {'Speedup':<12}")
            print("-" * 50)

            for sys_label, dims in systems:
                psi = make_state_vector(dims)
                key = jax.random.key(SEED + 100)

                def fn_orig():
                    r = qx.targeted_apply_kraus_map_trajectory(kraus_trunc, psi, key, subsystem)
                    r.data.block_until_ready()
                t_orig = bench(fn_orig) * 1000

                def fn_rdm():
                    r = qx.targeted_apply_kraus_map_trajectory_rdm(kraus_trunc, psi, key, subsystem)
                    r.data.block_until_ready()
                t_rdm = bench(fn_rdm) * 1000

                def fn_single():
                    r = qx.targeted_apply_kraus_map_single_trajectory(kraus_trunc, psi, key, subsystem)
                    r.data.block_until_ready()
                t_single = bench(fn_single) * 1000

                print(f"{sys_label:<8} {'original':<14} {t_orig:<12.3f} {'1.00x':<12}")
                print(f"{sys_label:<8} {'rdm':<14} {t_rdm:<12.3f} {f'{t_orig/t_rdm:.2f}x':<12}")
                print(f"{sys_label:<8} {'single':<14} {t_single:<12.3f} {f'{t_orig/t_single:.2f}x':<12}")
                print()


if __name__ == "__main__":
    main()
