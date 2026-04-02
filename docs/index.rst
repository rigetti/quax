Quax
====

**High-performance quantum information science with JAX**

Quax is a library for working with states, gates and superoperators in quantum information science. It's built on top of JAX, which enables great performance and automatic differentiation.

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

Key Features
------------

Quantum Objects
   States, gates and superoperators are represented as typed objects for clarity and ease of use

Standard Operators
   Composition, tensor products, scalar multiplication and powers are defined on all quantum objects allowing for natural manipulations.

Qudits
   Support for d-dimensional qudits, enabling operations beyond qubits

Batch Operations
   Operate on batches or ensembles of states for straightforward parallelization

Standard gate set
   A set of standard gates is included, based on the QUIL language specification

High Performance
   Built on JAX for GPU/TPU acceleration and automatic differentiation

Automatic Promotion
   Pure states and operators are automatically promoted to mixed states and superoperators when appropriate.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   installation
   quickstart
   quantum-objects
   examples/quax
   examples/hamiltonians
   examples/quantum-volume
   examples/leakage-randomized-benchmarking
   examples/qudits
   api/index
