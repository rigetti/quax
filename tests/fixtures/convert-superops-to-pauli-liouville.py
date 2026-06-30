#!/usr/bin/env python3

"""
Convert the superops/ to weyl_liouvilles/ and hermitian_weyl_liouvilles/.

This script requires TrueQ: https://trueq.quantumbenchmark.com/

It converts the column-stacked superoperators found in the superops directory to
Weyl-Liouvilles based on the unitary basis, and Hermitian Weyl-Liouvilles based on the
Hermitian basis. For d=2 these are the same, but for d>2 they differ.

These artifacts are used in testing.

To generate the superoperators, use the following command:

```bash
pytest tests/test_superoperator_transformations.py -k generate_superop_fixtures -q
python convert-superops-to-pauli-liouville.py quax/tests/fixtures/superops
```
"""

import argparse
import logging
from pathlib import Path
import trueq as tq  # type: ignore[import-untyped]
from functools import reduce
from operator import mul
import numpy as np


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load Superop matrices from an .npz file, convert them to "
            "Pauli-Liouville matrices, and save them to another .npz file."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input .npz file containing arrays of Choi matrices.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def row_to_col_superop(S: np.ndarray, d: int) -> np.ndarray:
    """
    Convert a superoperator from row-stacked convention to column-stacked.
    """
    if S.shape != (d * d, d * d):
        raise ValueError(f"Expected shape {(d * d, d * d)}, got {S.shape}")

    return S.reshape(d, d, d, d).transpose(1, 0, 3, 2).reshape(d * d, d * d)


def col_to_row_superop(S: np.ndarray, d: int) -> np.ndarray:
    """
    Convert a superoperator from column-stacked convention to row-stacked.
    """
    if S.shape != (d * d, d * d):
        raise ValueError(f"Expected shape {(d * d, d * d)}, got {S.shape}")

    return S.reshape(d, d, d, d).transpose(1, 0, 3, 2).reshape(d * d, d * d)


def superop_to_weyl_liouville(s: np.ndarray, qudit_dim: int, num_qudits: int):
    """
    Convert a column-stacked superoperator matrix to a Pauli-Liouville matrix.
    """
    d = qudit_dim**num_qudits
    s_row = col_to_row_superop(s, d)
    S = tq.math.Superop(s_row, dim=qudit_dim)
    # ptm_col = row_to_col_superop(S.ptm, d)
    # herm_ptm_col = row_to_col_superop(S.herm_ptm, d)
    return S.ptm, S.herm_ptm


def convert_arrays(data: np.lib.npyio.NpzFile):

    superops, ensemble_size, dims = (
        data["data"],
        data["ensemble_size"],
        data["dims"],
    )
    ensemble_size = tuple(ensemble_size)
    dims = tuple(int(d) for d in dims)

    # TrueQ only supports uniform qudit dimensions
    if len(set(dims)) != 1:
        raise ValueError(f"Mixed dims {dims} not supported by TrueQ. Skipping.")

    qudit_dim = dims[0]
    num_qudits = len(dims)

    logger.info(f"Converting {ensemble_size} superoperators, dims={dims}")

    # Reshape to 1D
    if ensemble_size:
        superops = superops.reshape((reduce(mul, ensemble_size), *superops.shape[-2:]))
    else:
        superops = superops.reshape((1,) + superops.shape[-2:])

    weyl_liouvilles, herm_weyl_liouvilles = [], []
    for s in superops:
        w, hw = superop_to_weyl_liouville(s, qudit_dim, num_qudits)
        weyl_liouvilles.append(w)
        herm_weyl_liouvilles.append(hw)

    weyl_liouvilles = np.asarray(weyl_liouvilles)
    herm_weyl_liouvilles = np.asarray(herm_weyl_liouvilles)

    # reshape to ensemble
    weyl_liouvilles = weyl_liouvilles.reshape(ensemble_size + weyl_liouvilles.shape[-2:])
    herm_weyl_liouvilles = herm_weyl_liouvilles.reshape(ensemble_size + herm_weyl_liouvilles.shape[-2:])

    return weyl_liouvilles, herm_weyl_liouvilles


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_dir = Path(args.input)
    weyl_dir = Path(args.input).parent / "weyl_liouvilles"
    herm_dir = Path(args.input).parent / "hermitian_weyl_liouvilles"

    weyl_dir.mkdir(parents=True, exist_ok=True)
    herm_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npz"))

    logger.info("Found %d files in %s", len(files), input_dir)

    for file in files:
        logger.info("Processing %s", file.name)

        data = np.load(file)

        try:
            weyl_liouvilles, herm_weyl_liouvilles = convert_arrays(data)
        except ValueError as e:
            logger.warning("Skipping %s: %s", file.name, e)
            continue

        weyl_data = dict(
            data=weyl_liouvilles,
            seed=data["seed"],
            dims=data["dims"],
            ensemble_size=data["ensemble_size"],
        )

        herm_weyl_data = dict(
            data=herm_weyl_liouvilles,
            seed=data["seed"],
            dims=data["dims"],
            ensemble_size=data["ensemble_size"],
        )

        weyl_output = weyl_dir / file.name.replace("superop_", "pauli-liouville_")
        herm_output = herm_dir / file.name.replace("superop_", "pauli-liouville_")

        logger.info("Saving Weyl-Liouville -> %s", weyl_output)
        np.savez(weyl_output, allow_pickle=True, **weyl_data)

        logger.info("Saving Hermitian Weyl-Liouville -> %s", herm_output)
        np.savez(herm_output, allow_pickle=True, **herm_weyl_data)

    logger.info("Done.")


if __name__ == "__main__":
    main()
