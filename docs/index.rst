Quax
====

**High-performance quantum information science with JAX**

Quax is a library for quantum operator transformations built on JAX, enabling hardware-accelerated
quantum computations with automatic differentiation.

----

Getting Started
---------------

.. grid:: 2

    .. grid-item-card:: Installation
        :link: installation
        :link-type: doc

        Install Quax using Poetry or pip

    .. grid-item-card:: Quickstart
        :link: quickstart
        :link-type: doc

        Learn the basics with examples

    .. grid-item-card:: API Reference
        :link: api/index
        :link-type: doc

        Detailed API documentation

    .. grid-item-card:: GitLab
        :link: https://gitlab.com/rigetti/qpu-hybrid-benchmark/quax

        Source code and issue tracker

Overview
--------

Quax provides JAX-based quantum operator transformations with support for:

* **Tensor-native data representation** - All quantum objects stored in tensor format for efficient operations
* **Multiple quantum state representations** - State vectors and density matrices
* **Multiple superoperator representations** - Kraus, Choi, Pauli-Liouville, and SuperOp
* **Quantum channel operations** - Compose and apply quantum channels
* **Distance metrics** - Fidelity calculations and process comparisons
* **Random state generation** - Create random quantum states and operators
* **Hardware acceleration** - GPU/TPU support via JAX

Key Features
------------

High Performance
   Built on JAX for GPU/TPU acceleration and automatic differentiation

Tensor-Native Storage
   Data stored in tensor format preserving qudit structure; matrix views available via ``.matrix`` property

Multiple Representations
   Seamlessly convert between different quantum operator representations

Composable Operations
   Chain quantum operations with intuitive Python syntax

Type-Safe
   Strong typing with dataclasses for quantum objects

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   installation
   quickstart
   api/index
