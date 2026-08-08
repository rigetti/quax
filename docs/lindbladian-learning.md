# Scalable Lindbladian Learning — design, evidence and reproduction guide

This note is written for someone building the real experiment. It covers the ansatz, the test battery,
the sampling scheme, the estimator (anchor states, derivative estimation, the local design), the
measured evidence, and the limits worth knowing before trusting a number.

Everything here is about the **scalable** algorithm: cost $O(n)$ in the register and $O(4^{k})$ in the
largest term weight $k$, with nothing larger than a $(k{+}1)$-qubit object ever constructed. A
finite-time global fit also exists in the module (`learn`); it is more noise-robust but exponential in
$n$, and is deliberately not documented here.

| Path | Contents |
|---|---|
| `src/quax/learning.py` | `Term`, `LindbladModel`, term constructors, `learn_local`, and the patch utilities (`rebuild_term`, `embed_operator`, `patch_model`, `neighbourhood`). Exported as `qx.learning`. |
| `tests/lindbladian_cases.py` | The 17-case battery and the `Experiment` wrapper (shot noise × turn-on ramp). |
| `tests/test_lindbladian_cases.py` | 112 tests validating the battery itself. |
| `tests/test_learning.py` | Tests of the learner, including the locality identity. |

Units throughout: **time in µs**, so dissipative rates are µs⁻¹ and coherent couplings are angular
frequencies in rad·µs⁻¹ (i.e. $2\pi f$; 1 MHz $= 2\pi$ rad·µs⁻¹).

---

## 1. The ansatz

A `LindbladModel` is a **linear** ansatz — a fixed list of unit generators with a coefficient vector:

$$\mathcal{L}(\theta, t) = \sum_j \theta_j\left[\cos(\omega_j t)\,G_j + \sin(\omega_j t)\,Q_j\right]$$

Static terms have $\omega_j = 0$ and $Q_j = 0$. Linearity in $\theta$ is the property the whole method
rests on: it makes the estimator a single convex least-squares solve.

Each `Term` records three things that matter:

* `support` — the qubits it acts on. This is what makes local fitting possible.
* `coherent` — Hamiltonian (coefficient may take either sign) vs dissipator (coefficient is a rate,
  constrained non-negative).
* `local_hamiltonian` / `local_jump` / `local_quadrature` — the operator **restricted to its own
  support**. This is what makes the estimator scalable: a term can be rebuilt on any small register
  containing its support (`rebuild_term`) without ever forming a full-system operator.

Conventions for the coefficients:

| Constructor | Physics | Coefficient means |
|---|---|---|
| `amplitude_damping_term(q, n)` | $D[\sigma^-_q]$ | $1/T_1$ |
| `excitation_term(q, n)` | $D[\sigma^+_q]$ | thermal excitation rate |
| `dephasing_term(q, n)` | $D[Z_q/\sqrt2]$ | $1/T_\varphi$ |
| `joint_decay_term(i, j, n)` | $D[\sigma^-_i + \sigma^-_j]$ | collective decay rate |
| `detuning_term(q, n)` | $H = Z_q/2$ | angular detuning $\delta$ |
| `drive_term(q, n)` | $H = X_q/2$ | Rabi rate $\Omega$ |
| `zz_term(i, j, n)` | $H = Z_iZ_j$ | ZZ coupling |
| `xz_term(i, j, n)` | $H = X_iZ_j/2$ | drive crosstalk |
| `exchange_term(i, j, n, frequency)` | $H = \tfrac12(X_iX_j + Y_iY_j)$ | exchange $g$; `frequency` $=\Delta_{ij}$ makes it time dependent |

A new term type needs only its local operator and its support; everything downstream is generic.

---

## 2. The test battery

17 cases progressing 1q → 2q → 4q, and decoherence → detuning → ZZ → drives → time-dependent XY, plus
two exotic cases (collective decay, XZ crosstalk). Each supplies a structurally correct ansatz, random
true coefficients from realistic ranges, and an initial guess within **50%** of the truth.

Measured properties (seed 7):

* dissipative rates **0.027–0.066 µs⁻¹** ($T_1$, $T_\varphi$ of 15–60 µs);
* coherent couplings up to **72 rad·µs⁻¹** (Rabi periods 80–200 ns);
* **coherent/dissipative ratios 69–1290** — the regime that makes this hard;
* guess errors 20–50%.

The battery's own tests check structural consistency, **identifiability** (unit generators must be
linearly independent, else coefficients are not recoverable even in principle), CPTP evolution, the
guess tolerance, the shot-noise magnitude, and that the ramp corrupts the initial states while keeping
them informationally complete.

> Writing these tests first paid for itself immediately: the scale-separation test caught a bug where
> the MHz→rad·µs⁻¹ constant was **1000× too small**, which had made every "fast" coherent term slower
> than the decoherence.

---

## 3. The sampling scheme

**Prepare a random Pauli eigenstate on every qubit each shot; measure a random Pauli basis on every
qubit.** One such dataset serves every term simultaneously — that is the key practical economy.

Why it works: averaging a qubit over its six Pauli eigenstates is exactly $\mathbb{1}/2$. So binning
the shots by the preparation on a support $T$, and averaging over everything else, realises the reduced
experiment with inputs $\rho_T \otimes (\mathbb{1}/2)^{\otimes\mathrm{env}}$ and observables
$O_T \otimes \mathbb{1}$ — without ever deliberately preparing a maximally mixed state.

In this codebase `measure(times)` returns the complete $6^n \times 4^n$ grid and
`_marginalise_data(data, support, n)` performs that binning; verified to reproduce a *directly
constructed* reduced experiment to **3e-16**, including when a ramp has already acted.

**Settings.** For support $T$ of weight $k$ the reduced experiment has $6^k$ preparations $\times\ 4^k$
observables. Summed over supports that is

$$\sum_T 6^{|T|}4^{|T|} \;=\; 24\,n + 576\,|E| \quad\text{(1- and 2-local terms)},$$

**linear in the number of terms**, against $24^n$ for complete tomography — a 2654× reduction already
at $n=5$, growing ~19× per added qubit. An $n=10$ device needs ~6,000 local settings instead of
$\sim 6\times10^{13}$.

**Shot budget.** A shot is usable for a weight-$k$ support when its measurement basis matches
(probability $3^{-k}$), and it lands in one of $6^k$ preparation bins. To accumulate $M$ shots per
(preparation, basis) cell for the largest weight $k$:

$$N_{\text{total}} \;\approx\; M \cdot 18^{k}$$

so $324\,M$ for two-local terms. With $M = 10^5$ that is $\sim3\times10^7$ shots — for the whole
device, not per term.

---

## 4. The estimator

### 4.1 The locality identity — why no light cone appears

With every qubit outside $T$ maximally mixed, the short-time derivative of a $T$-local observable is
**exactly**

$$\frac{d}{dt}\langle O_T\rangle \;=\; \mathrm{Tr}\!\left[O_T\,\mathcal{L}_T(\rho_T)\right],$$

where $\mathcal{L}_T$ is the reduction of the generator onto $T$. A term straddling the boundary
contributes **nothing** — its factor outside $T$ is a traceless Pauli and
$\mathrm{Tr}[(\mathbb{1}/2)P] = 0$ — and a term disjoint from $T$ cancels identically.

Verified against exact global evolution on a 5-qubit ring:

| coupling | \|global $d/dt$ − support-only $d/dt$\| |
|---|---|
| ZZ | **2.2e-16** |
| XY exchange | **1.8e-15** |

Machine precision **including the spreading transverse coupling**. This is the crucial difference from
finite-time methods: $e^{\tau\mathcal{L}}$ sums all orders and spreads support globally (forcing a
light-cone/patch approximation), whereas the derivative is *exactly* local at any coupling strength.

> **Collective jumps need care.** Restricting to terms whose support lies *inside* $T$ is wrong for a
> sum-type jump operator: expanding $D[\sigma^-_i + \sigma^-_j]$ gives
> $D[\sigma^-_i] + D[\sigma^-_j] + \text{cross}$. The cross terms are traceless and do vanish, but
> $D[\sigma^-_i]$ lives entirely inside a *single-qubit* support. Every **overlapping** term must
> therefore be reduced onto $T$, evaluated on the small register $S_j \cup T$. Getting this wrong cost
> 20.7% on `2q-joint-decay`; fixing it gave **0.00002%**.

### 4.2 Anchor states — do not assume what you prepared

The identity needs the *actual* state at the reference time. A turn-on ramp rotates the prepared
states, so the estimator reconstructs them from the data by linear inversion of the marginal,

$$\hat\rho_k^{(T)} \;=\; \frac{1}{2^{|T|}}\sum_a \langle O_a\rangle_k\,O_a ,$$

and uses $\hat\rho_k^{(T)}$ in the design. This matters enormously: using the *intended* states instead
gave 16,500% error on the driven single-qubit case under a 10 ns ramp; using the measured ones gives
**0.0064%** (§7).

### 4.3 Derivative estimation

Sample $p$ closely spaced times ($p = 5$ by default) starting at the reference time, and fit a
degree-$d$ least-squares polynomial ($d = 2$) to each local expectation; the linear coefficient is the
slope. A polynomial fit rather than a raw finite-difference stencil — the degree is a bias/variance
knob, and low degree is markedly more noise-robust.

**Step size** is the critical parameter. It must be small against the fastest timescale in the
generator, coherent *or* dissipative:

```
step = 0.002 / max(|coherent coefficients|, |dissipative rates|, |modulation frequencies|)
```

Truncation leakage measured against the exact slope (5-qubit ring, support (0,1)):

| coupling | dt = 0.5 ns | 2 ns | 10 ns | 50 ns |
|---|---|---|---|---|
| ZZ | 4.8e-12 | 1.2e-09 | 7.6e-07 | 4.1e-04 |
| XY exchange | 2.7e-08 | 6.7e-06 | 7.7e-03 | 9.3e-01 |

The leading order is exactly local; the residual is second-order leakage from neighbours and is
controlled purely by the step.

### 4.4 The design matrix and the solve

For each support $T$, and each term $j$ overlapping $T$:

1. build the small register $U = S_j \cup T$ (never larger than $k + |T|$ qubits);
2. rebuild term $j$ on $U$ from its local operators (`rebuild_term`);
3. embed the measured anchor states as $\hat\rho^{(T)}_k \otimes (\mathbb{1}/2)^{U\setminus T}$ and the
   observables as $O^{(T)}_a \otimes \mathbb{1}$;
4. the design entry is $A_{(k,a),\,j} = \mathrm{Tr}\!\left[O_a\,\mathcal{L}_j(\hat\rho_k)\right]$.

Stack the blocks over all supports into one system $A\theta = g$ (with $g$ the measured slopes) and
solve by least squares, clipping dissipative coefficients at zero. Because $\mathcal{L}$ is linear in
$\theta$ this is **convex** — one solve, no optimiser, no initial-guess sensitivity, no local minima.
The initial guess is used *only* to choose the derivative step.

---

## 5. Results

Worst-case relative error over all coefficients, noiseless, from 20–50% initial guesses:

| case | qubits | error |
|---|---|---|
| `1q-t1-t2` | 1 | 2.5e-6% |
| `1q-thermal` | 1 | 1.0e-6% |
| `1q-driven-detuned` | 1 | 6.4e-3% |
| `2q-detuning-zz` | 2 | 7.3e-3% |
| `2q-driven-zz` | 2 | 9.4e-3% |
| `2q-joint-decay` | 2 | 2.3e-5% |
| `2q-xz-crosstalk` | 2 | 2.7e-3% |
| `2q-exchange-resonant` | 2 | 1.8e-3% |
| `4q-detuning-zz-ring` | 4 | 5.9e-3% |
| `4q-driven-ring` | 4 | 2.0e-3% |

Every static case is recovered essentially exactly, including a Rabi drive ~1300× faster than the
decoherence it sits alongside.

---

## 6. Scaling

Design construction from local operator data alone, ring lattice, $4n$ terms:

| n | terms | design rows | design cols | largest register | largest dim | build (s) | solve (s) |
|---|---|---|---|---|---|---|---|
| 8 | 32 | 4,800 | 32 | 3 | 64 | 0.32 | 0.11 |
| 16 | 64 | 9,600 | 64 | 3 | 64 | 0.58 | 0.20 |
| 32 | 128 | 19,200 | 128 | 3 | 64 | 1.02 | 0.16 |
| **64** | 256 | 38,400 | 256 | **3** | **64** | 1.99 | 0.43 |

Rows and columns grow **linearly**; the largest object stays a **3-qubit register** whatever the device
size. Sixty-four qubits takes ~2.4 s end to end. The cost is $O(n)$ in the register and $O(4^{k})$ in
the largest term weight — nothing else.

One implementation caveat: `Term`/`LindbladModel` currently materialise full-space $4^n\times4^n$
generators eagerly, which is needed only by the simulation that *generates* test data. It caps model
construction at $n \approx 6$. The estimator itself never needs them — the table above is built purely
from local data — so making those matrices lazy is the one change required to run `learn_local` at
device scale.

---

## 7. Limits to know before trusting a number

**Shot noise is the binding constraint.** The derivative step must be small against the fastest term
($\sim 0.002/\Omega$), so noise is amplified by $1/\delta t$. At $10^5$ shots per setting the local
estimator is far worse than a finite-time global fit. Larger steps with higher-order stencils recover
some of it (`1q-t1-t2`: 368% at 2 ns → 4% at 1 µs) but cannot resolve a fast drive at coarse spacing.
**This is the main open problem: a scalable *and* noise-robust estimator.** Promising directions —
inverse-variance weighting (shot noise is binomial, and reconstructed quantities have correlated
errors), regularisation toward the initial guess, and Tikhonov/total-variation regularised
differentiation instead of a plain polynomial fit.

**Turn-on ramp: single-qubit terms are free, two-qubit terms are not.** Measured with a realistic ramp
(Hamiltonian rising linearly to full strength while decoherence acts throughout,
`ramp_profile="coherent"`):

| case | no ramp | 10 ns | 20 ns | 50 ns | 200 ns |
|---|---|---|---|---|---|
| `1q-driven-detuned` | 0.0064% | 0.0064% | 0.0064% | 0.0064% | 0.0064% |
| `2q-detuning-zz` | 0.007% | 0.61% | 1.2% | 3.0% | 12.1% |
| `2q-driven-zz` | 0.009% | 2.5% | 4.7% | 7.3% | 23.1% |
| `2q-exchange-resonant` | 0.002% | 66% | 132% | 319% | 900% |

A ramp built from **single-qubit** Hamiltonian terms is **exactly harmless**: a local unitary maps
$\mathbb{1}/2 \mapsto \mathbb{1}/2$, so the environment stays maximally mixed and the identity is
untouched. Only **two-qubit coherent** ramp terms hurt, by correlating the support with its
environment, with error growing linearly in (coupling × duration). For a 20 ns ramp on a device whose
crosstalk is ZZ at the ~100 kHz scale the cost is **1–5%** — a tolerable systematic. Strong (MHz)
transverse exchange during the ramp is not. Since the error is linear in the ramp-induced phase,
reconstructing the reference state on the support *plus its neighbours* (a radius-1 patch of the
**state**, still $O(1)$) should absorb the leading correction.

**Weakly-constrained rates.** A rate the data barely constrains can be pushed to zero. In the convex
local estimator this appears as clipping at the non-negativity bound rather than as divergence, but any
rate that comes back near zero deserves its own dedicated long-baseline measurement.

**Time-dependent (modulated) generators** are not yet fittable: the time-ordered propagator splits
$t = nT + r$ and uses `matrix_power`, needing a concrete cycle count, so it cannot be differentiated
through. The corresponding test is marked `xfail`.

---

## 8. Reproducing

```bash
cd quax
JAX_ENABLE_X64=1 poetry run pytest tests/test_lindbladian_cases.py -q   # validate the battery
JAX_ENABLE_X64=1 poetry run pytest tests/test_learning.py -q            # test the learner
JAX_ENABLE_X64=1 poetry run pytest tests/test_learning.py -q -m slow    # 4q + time-dependent
```

64-bit precision is **required** — the schedule spans nanoseconds to tens of microseconds.

```python
from quax.learning import LindbladModel, amplitude_damping_term, dephasing_term, drive_term, learn_local

model = LindbladModel(1, (amplitude_damping_term(0, 1), dephasing_term(0, 1), drive_term(0, 1)))
result = learn_local(model, experiment.measure, initial_guess)
print(dict(zip(model.names, result.coefficients)))
```

`measure(times)` must return `(len(times), 6**n, 4**n)` expectations — the complete Pauli tomography at
the requested times. In a real experiment this is the marginalised random-Pauli dataset of §3. Keep
`settling = 0`: the locality identity relies on the environment being maximally mixed at the reference
time.

---

## 9. A note on method

Every substantive fix came from a *diagnostic* that separated competing explanations, not from
parameter tuning:

* comparing the loss at the fit against the loss at the truth distinguished an optimisation failure
  from an information limit — the highest-value single measurement in the project;
* running the same data through an inverting estimator and a forward model isolated noise
  amplification;
* toggling the ramp independently of the noise showed anchoring was already correct, so the remaining
  failures were entirely about noise;
* testing the patch idea on a ZZ-only model gave a misleadingly perfect result; repeating it with a
  *spreading* XY coupling exposed the light cone the diagonal model had hidden.

Parameter sweeps moved nothing until the underlying mechanism was identified. Two of the most useful
results were negative: co-fitting the post-ramp state as a nuisance parameter *cannot* work (the extra
freedom absorbs the signal that pins the drive frequency), and restricting local fits by "support
inside $T$" *silently* breaks collective jump operators.
