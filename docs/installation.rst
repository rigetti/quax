Installation
============

Requirements
------------

- Python >= 3.12
- JAX >= 0.8.2

Gitlab Access Token
--------------------

To access the package, you need to have your gitlab access token set up.

Get it here: `GitLab Access Tokens <https://gitlab.com/-/user_settings/personal_access_tokens>`_

Install with Poetry
-------------------

.. code-block:: bash

   poetry add rigetti-quax

Or install from source:

.. code-block:: bash

   git clone https://gitlab.com/rigetti/qpu-hybrid-benchmark/quax.git
   cd quax
   poetry install

Development Installation
------------------------

To install with development dependencies:

.. code-block:: bash

   poetry install --with dev

This includes:

- pytest for testing
- ruff for linting
- qutip for quantum information comparisons
- sphinx for documentation

Verify Installation
-------------------

.. code-block:: python

   import quax as qx
   
   # Create a simple quantum state
   state = qx.zero_state_vector(dims=(2,))
   print(state)
