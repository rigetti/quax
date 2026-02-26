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

- **Quantum Objects**: States, gates and superoperators objects are defined ot allow natural manipulations and operations.
- **Standard operators** composition/application `@`, tensor products `|`, scalar multiplication `*` and powers `*` are defined on all quantum objects.
- **Qudits** Operations on d-dimensional qudits are supported.
- **Batch operations** Operating on batches or ensembles of states is supported for straightforward parallelization.

Key Features
------------

High Performance
   Built on JAX for GPU/TPU acceleration and automatic differentiation

Multiple Representations
   Seamlessly convert between different quantum operator representations

Composable Operations
   Chain quantum operations with intuitive Python syntax

Type-Safe
   Standard objects such as Unitaries, Chois and Density matrices are all typed, clarifying the nature of various objects.

Automatic Promotion
   Pure states and operators are automatically promoted to mixed states and superoperators when appropriate.



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   installation
   quickstart
   quantum_objects
   reference
   api/index
