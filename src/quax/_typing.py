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
"""Type variables shared across the package."""

from typing import TypeVar

import jax
import numpy as np
import numpy.typing as npt

ArrayT = TypeVar("ArrayT", npt.NDArray[np.number], jax.Array)
"""Either a numeric NumPy array or a JAX array.

Functions can use this to express support for both NumPy and JAX arrays while enabling the type
checker to verify that the output is the same kind of array as the input. It constrains the
container kind only, not the dtype.
"""
