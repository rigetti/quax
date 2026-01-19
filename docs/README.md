# Quax Documentation

This directory contains the documentation for Quax.

## Building the Documentation

To build the documentation, you need to install the documentation dependencies:

```bash
poetry install --with dev
```

Then build the HTML documentation:

```bash
cd docs
poetry run make html
```

The built documentation will be in `_build/html/`. You can open `_build/html/index.html` in your browser.

## Documentation Style

Quax uses reStructuredText (ReST) style docstrings and Sphinx with the Furo theme (same as JAX).

The documentation uses:
- **Furo theme** - Modern, clean theme used by JAX
- **sphinx-design** - For grid layouts and cards
- **sphinx-copybutton** - Copy button for code blocks
- **Napoleon** - Support for both NumPy and Google style docstrings

Example:

```python
def my_function(param1, param2):
    """
    Short description of the function.

    Longer description of the function explaining what it does,
    its purpose, and any important details.

    :param param1: Description of param1.
    :type param1: type
    :param param2: Description of param2.
    :type param2: type
    :return: Description of the return value.
    :rtype: return_type

    Example::

        >>> my_function(1, 2)
        3
    """
    return param1 + param2
```
