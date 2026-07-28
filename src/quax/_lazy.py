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

"""Lazy, precision-aware module constants (PEP 562).

Constant quantum objects such as ``gates.X`` or ``states.KET0`` used to be built at import time.
Because JAX resolves ``complex`` to ``complex64`` or ``complex128`` depending on the ``jax_enable_x64``
setting *at the moment the array is created*, an import-time constant permanently baked in whatever
precision happened to be active when ``quax`` was first imported.  If a user enabled x64 afterwards,
the constants kept the wrong dtype.

This helper turns a registry of zero-argument *builders* into a module-level ``__getattr__`` /
``__dir__`` pair so that each constant is built on first access — at the *current* precision — and
cached per canonical complex dtype.  Attribute access (``qx.gates.X``) is preserved; the constants
never become functions the caller has to invoke.
"""

from functools import lru_cache
from typing import Any, Callable

import jax
import jax.numpy as jnp


def make_lazy_getter(builders: dict[str, Callable[[], Any]]) -> Callable[[str], Any]:
    """Build a precision-aware getter for a registry of lazily-built constants.

    The returned ``get(name)`` constructs ``builders[name]()`` on first access and caches the result
    keyed by the current canonical complex dtype, so a change to ``jax_enable_x64`` after import
    yields a fresh, correctly-typed instance.  Builders may call ``get`` to obtain their dependencies
    at the same precision.  ``get`` raises :class:`AttributeError` for unknown names so a module's
    ``__getattr__`` can delegate to it directly.

    Each module wires this up with an explicit module-level ``__getattr__`` / ``__dir__`` (rather
    than assigning them dynamically) so static type checkers recognise the lazy attributes.
    """

    @lru_cache(maxsize=None)
    def _build(name: str, _complex_dtype: Any) -> Any:
        # _complex_dtype is part of the cache key so a precision change yields a fresh instance.
        # ensure_compile_time_eval forces concrete evaluation even when a constant is first accessed
        # inside a jit trace, so the cached array is never a leaked tracer.
        with jax.ensure_compile_time_eval():
            return builders[name]()

    def get(name: str) -> Any:
        if name not in builders:
            raise AttributeError(name)
        return _build(name, jax.dtypes.canonicalize_dtype(jnp.complex128))

    return get
