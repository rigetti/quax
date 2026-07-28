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

"""Cold-import benchmarks for ``quax``.

Run with:
    make benchmark

These benchmarks are NOT included in the default test suite (``make test-package``).

Why is ``import quax`` slow, and what do these benchmarks measure?
------------------------------------------------------------------
``import quax`` transitively imports JAX and initialises its XLA backend.  That JAX/XLA startup is
by far the dominant, effectively fixed cost of importing the package -- ``test_import_jax_baseline``
measures it so the quax-attributable overhead (``test_import_quax`` minus the baseline) is visible.
An ``-X importtime`` profile confirms JAX (and its ``lax``/``numpy`` subpackages) as the largest
single contributor, followed by ``quax`` eagerly importing its own module graph and -- when the
optional ``plot`` extra is installed -- ``plotly`` via :mod:`quax._visualization`.

Historically a further cost was eager, module-level *construction* of quantum objects at import: the
``gates``/``states``/``ensembles`` constants were all built when the module was first imported,
including the (comparatively expensive) icosahedral-group generation in ``ensembles`` and the
``expm``-based ``ECR``/``SYCAMORE`` gates.  Those constants are now built lazily on first access
(see :mod:`quax._lazy`), so that construction no longer runs on the import path.
``test_first_gate_access`` measures importing plus touching one lazy constant, to show that
first-access cost is now paid on demand rather than up front (and only for what is actually used).

Each benchmark spawns a fresh subprocess per round: an in-process ``import`` is cached after the
first call, so measuring a cold import requires a new interpreter each time.  This includes Python
interpreter startup, which is common to every case and cancels out when comparing them.
"""

import subprocess
import sys

import pytest

# Subprocess imports are seconds-scale, so cap the round count rather than letting pytest-benchmark
# auto-calibrate with many iterations.
_PEDANTIC = {"rounds": 10, "iterations": 1, "warmup_rounds": 1}


def _run(code: str) -> None:
    """Run ``code`` in a fresh interpreter, raising if it exits non-zero."""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.benchmark(group="import")
def test_import_jax_baseline(benchmark):
    """Baseline: cold ``import jax`` (JAX + XLA backend init), the dominant fixed cost."""
    benchmark.pedantic(lambda: _run("import jax"), **_PEDANTIC)


@pytest.mark.benchmark(group="import")
def test_import_quax(benchmark):
    """Cold ``import quax``; subtract the jax baseline for quax's own import overhead."""
    benchmark.pedantic(lambda: _run("import quax"), **_PEDANTIC)


@pytest.mark.benchmark(group="import")
def test_first_gate_access(benchmark):
    """Cold ``import quax`` plus one lazy-constant access, showing on-demand construction cost."""
    benchmark.pedantic(lambda: _run("import quax; quax.gates.X"), **_PEDANTIC)
