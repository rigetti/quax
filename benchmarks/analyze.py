#!/usr/bin/env python3
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

"""Analyze pytest-benchmark JSON output and produce plots + markdown report.

Usage:
    python benchmarks/analyze.py benchmarks/results/64bit.json
    python benchmarks/analyze.py benchmarks/results/*.json

Produces:
    benchmarks/results/report.md   — Markdown with embedded plot images
    benchmarks/results/*.png       — Individual plot images
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Rigetti standard colours
# ---------------------------------------------------------------------------

_COLORS = [
    "#00B5AD",  # teal
    "#EF476F",  # pink
    "#118AB2",  # blue
    "#073B4C",  # dark blue
    "#FFD166",  # yellow
    "#06D6A0",  # green
    "#8338EC",  # purple
    "#FB5607",  # orange
]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


# ---------------------------------------------------------------------------
# Dimension string → Hilbert-space dimension
# ---------------------------------------------------------------------------

# Map system labels used in test IDs back to per-qudit dimensions
_SYSTEM_DIMS: dict[str, tuple[int, ...]] = {
    "2Q": (2, 2),
    "4Q": (2,) * 4,
    "6Q": (2,) * 6,
    "8Q": (2,) * 8,
    "10Q": (2,) * 10,
    "16Q": (2,) * 16,
    "20Q": (2,) * 20,
}


def _num_qubits(system_label: str) -> int:
    dims = _SYSTEM_DIMS.get(system_label)
    if dims is None:
        return 0
    return len(dims)


# ---------------------------------------------------------------------------
# Parse test ID fields
# ---------------------------------------------------------------------------

# Patterns for the various test parameter ID formats:
# test_targeted_apply_unitary[1Q-ens()-1Qop]
# test_kraus_trajectory[1Q-rank1-full-keys1]   (no ensemble)
# test_instrument_dm[1Q-ens()-ideal]

_RE_UNITARY = re.compile(r"test_targeted_apply_unitary\[(?P<sys>\w+)-ens(?P<ens>\([^)]*\))-(?P<sub>\w+)\]")
_RE_SUPEROP = re.compile(r"test_targeted_apply_superop\[(?P<sys>\w+)-ens(?P<ens>\([^)]*\))-(?P<sub>\w+)\]")
_RE_KRAUS = re.compile(
    r"test_kraus_trajectory\[(?P<sys>\w+)-ens(?P<ens>\([^)]*\))-(?P<src>rank\d+|depolarizing)-(?P<trunc>full|trunc)\]"
)
_RE_INST_DM = re.compile(r"test_instrument_dm\[(?P<sys>\w+)-ens(?P<ens>\([^)]*\))-(?P<itype>ideal|noisy)\]")
_RE_INST_SV = re.compile(r"test_instrument_sv\[(?P<sys>\w+)-ens(?P<ens>\([^)]*\))-(?P<itype>ideal|noisy)\]")


def _classify(bench: dict) -> dict | None:
    """Parse a benchmark dict and return a record with extracted fields, or None."""
    name = bench["name"]
    mean_us = bench["stats"]["mean"] * 1e6  # seconds → µs
    median_us = bench["stats"]["median"] * 1e6
    stddev_us = bench["stats"]["stddev"] * 1e6

    for regex, category in [
        (_RE_UNITARY, "unitary_sv"),
        (_RE_SUPEROP, "superop_dm"),
        (_RE_KRAUS, "kraus_sv"),
        (_RE_INST_DM, "instrument_dm"),
        (_RE_INST_SV, "instrument_sv"),
    ]:
        m = regex.search(name)
        if m:
            rec: dict = {
                "category": category,
                "system": m.group("sys"),
                "num_qubits": _num_qubits(m.group("sys")),
                "mean_us": mean_us,
                "median_us": median_us,
                "stddev_us": stddev_us,
                "name": name,
            }
            if "ens" in m.groupdict():
                rec["ensemble"] = m.group("ens")
            if category == "kraus_sv":
                rec["source"] = m.group("src")
                rec["truncated"] = m.group("trunc")
            if category in ("instrument_dm", "instrument_sv"):
                rec["inst_type"] = m.group("itype")
            if category in ("unitary_sv", "superop_dm"):
                rec["subsystem"] = m.group("sub")
            return rec
    return None


# ---------------------------------------------------------------------------
# Load + merge JSON results
# ---------------------------------------------------------------------------


def _load_results(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        path = Path(p)
        precision = "64bit" if "64" in path.stem else "32bit"
        with open(path) as f:
            data = json.load(f)
        for bench in data.get("benchmarks", []):
            rec = _classify(bench)
            if rec:
                rec["precision"] = precision
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _base_layout(title: str, yaxis_title: str = "Mean time (µs)") -> dict:
    return dict(
        title=title,
        xaxis_title="Number of qubits",
        yaxis_title=yaxis_title,
        yaxis_type="log",
        template="plotly_white",
        width=1000,
        height=600,
        legend=dict(font=dict(size=10)),
    )


def _add_line(
    fig: go.Figure,
    recs: list[dict],
    label: str,
    color: str,
    dash: str = "solid",
    symbol: str = "circle",
) -> None:
    """Add a single connected trace from a list of records, sorted by num_qubits."""
    if not recs:
        return
    recs = sorted(recs, key=lambda r: r["num_qubits"])
    fig.add_trace(
        go.Scatter(
            x=[r["num_qubits"] for r in recs],
            y=[r["mean_us"] for r in recs],
            error_y=dict(type="data", array=[r["stddev_us"] for r in recs], visible=True),
            mode="markers+lines",
            marker=dict(size=7, color=color, symbol=symbol),
            line=dict(color=color, dash=dash, width=2),
            name=label,
        )
    )


def _plot_unitary_or_superop(records: list[dict], category: str, subsystem: str, title: str) -> go.Figure:
    """X = num qubits, one line per ensemble size. Filtered to a single op size."""
    fig = go.Figure()
    cat_recs = [r for r in records if r["category"] == category and r.get("subsystem") == subsystem]
    if not cat_recs:
        return fig

    ensembles = sorted({r["ensemble"] for r in cat_recs})
    for i, ens in enumerate(ensembles):
        subset = [r for r in cat_recs if r["ensemble"] == ens]
        label = f"ens={ens}"
        _add_line(fig, subset, label, _color(i))

    fig.update_layout(**_base_layout(title))
    return fig


def _plot_kraus_trunc_effect(records: list[dict], title: str) -> go.Figure:
    """Show truncation effect: full vs trunc for each source, no ensemble."""
    fig = go.Figure()
    cat_recs = [r for r in records if r["category"] == "kraus_sv" and r.get("ensemble") == "()"]
    if not cat_recs:
        return fig

    sources = sorted({r["source"] for r in cat_recs})
    for i, src in enumerate(sources):
        for trunc in ["full", "trunc"]:
            subset = [r for r in cat_recs if r["source"] == src and r["truncated"] == trunc]
            dash = "dash" if trunc == "trunc" else "solid"
            label = f"{src} ({trunc})"
            _add_line(fig, subset, label, _color(i), dash=dash)

    fig.update_layout(**_base_layout(title))
    return fig


def _plot_kraus_by_ensemble(records: list[dict], title: str) -> go.Figure:
    """One line per ensemble size, fix source=rank1 full."""
    fig = go.Figure()
    cat_recs = [
        r for r in records
        if r["category"] == "kraus_sv" and r["source"] == "rank1" and r["truncated"] == "full"
    ]
    if not cat_recs:
        return fig

    ensembles = sorted({r["ensemble"] for r in cat_recs})
    for i, ens in enumerate(ensembles):
        subset = [r for r in cat_recs if r["ensemble"] == ens]
        label = f"ens={ens}"
        _add_line(fig, subset, label, _color(i))

    fig.update_layout(**_base_layout(title))
    return fig


def _plot_instrument(records: list[dict], category: str, title: str) -> go.Figure:
    """One line per (ensemble, inst_type).

    Color = ensemble, symbol = ideal/noisy.
    """
    fig = go.Figure()
    cat_recs = [r for r in records if r["category"] == category]
    if not cat_recs:
        return fig

    ensembles = sorted({r["ensemble"] for r in cat_recs})
    for i, ens in enumerate(ensembles):
        for itype in ["ideal", "noisy"]:
            subset = [r for r in cat_recs if r["ensemble"] == ens and r["inst_type"] == itype]
            dash = "dash" if itype == "noisy" else "solid"
            symbol = "x" if itype == "noisy" else "circle"
            label = f"ens={ens}  {itype}"
            _add_line(fig, subset, label, _color(i), dash=dash, symbol=symbol)

    fig.update_layout(**_base_layout(title))
    return fig


def _plot_precision_comparison(
    records: list[dict], category: str, title: str, filter_fn=None,
) -> go.Figure:
    """Compare 64-bit vs 32-bit for a category. One line per precision."""
    fig = go.Figure()
    cat_recs = [r for r in records if r["category"] == category]
    if filter_fn:
        cat_recs = [r for r in cat_recs if filter_fn(r)]
    if not cat_recs:
        return fig

    precisions = sorted({r["precision"] for r in cat_recs})
    for i, prec in enumerate(precisions):
        subset = [r for r in cat_recs if r["precision"] == prec]
        if subset:
            _add_line(fig, subset, prec, _color(i))

    fig.update_layout(**_base_layout(title))
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _summary_table(records: list[dict], category: str) -> str:
    """Build a markdown table summarizing benchmarks for a category."""
    rows = [r for r in records if r["category"] == category]
    if not rows:
        return "_No data._\n"

    rows.sort(key=lambda r: (r.get("precision", ""), r["num_qubits"], r.get("ensemble", "")))

    lines = ["| System | Qubits | Ensemble | Mean (µs) | Std (µs) | Precision |"]
    lines.append("|--------|--------|----------|-----------|----------|-----------|")
    for r in rows:
        ens = r.get("ensemble", "-")
        lines.append(
            f"| {r['system']} | {r['num_qubits']} | {ens} "
            f"| {r['mean_us']:.1f} | {r['stddev_us']:.1f} | {r.get('precision', '?')} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(json_paths: list[str], output_dir: str = "benchmarks/results") -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = _load_results(json_paths)
    if not records:
        print("No benchmark records found.", file=sys.stderr)
        sys.exit(1)

    # Split by precision for separate plots
    by_precision: dict[str, list[dict]] = {}
    for r in records:
        by_precision.setdefault(r.get("precision", "unknown"), []).append(r)

    plot_files: list[tuple[str, str]] = []  # (title, relative_path)

    for prec, recs in sorted(by_precision.items()):
        sfx = f"_{prec}"

        # --- Unitary (one plot per op size) ---
        for sub in ["1Qop", "2Qop", "3Qop"]:
            fig = _plot_unitary_or_superop(recs, "unitary_sv", sub, f"Unitary apply (SV) — {sub} [{prec}]")
            if fig.data:
                fname = f"unitary_sv_{sub}{sfx}.png"
                pio.write_image(fig, str(outdir / fname), scale=2)
                plot_files.append((f"Unitary apply (SV) — {sub} [{prec}]", fname))

        # --- SuperOp (one plot per op size) ---
        for sub in ["1Qop", "2Qop", "3Qop"]:
            fig = _plot_unitary_or_superop(recs, "superop_dm", sub, f"SuperOp apply (DM) — {sub} [{prec}]")
            if fig.data:
                fname = f"superop_dm_{sub}{sfx}.png"
                pio.write_image(fig, str(outdir / fname), scale=2)
                plot_files.append((f"SuperOp apply (DM) — {sub} [{prec}]", fname))

        # --- Kraus: source & truncation (no ensemble) ---
        fig = _plot_kraus_trunc_effect(recs, f"Kraus trajectory: source & truncation [{prec}]")
        if fig.data:
            fname = f"kraus_source{sfx}.png"
            pio.write_image(fig, str(outdir / fname), scale=2)
            plot_files.append((f"Kraus trajectory: source & truncation [{prec}]", fname))

        # --- Kraus: ensemble scaling (rank1 full) ---
        fig = _plot_kraus_by_ensemble(recs, f"Kraus trajectory: ensemble scaling (rank1 full) [{prec}]")
        if fig.data:
            fname = f"kraus_ensemble{sfx}.png"
            pio.write_image(fig, str(outdir / fname), scale=2)
            plot_files.append((f"Kraus trajectory: ensemble scaling [{prec}]", fname))

        # --- Instrument DM ---
        fig = _plot_instrument(recs, "instrument_dm", f"Instrument apply (DM) [{prec}]")
        if fig.data:
            fname = f"instrument_dm{sfx}.png"
            pio.write_image(fig, str(outdir / fname), scale=2)
            plot_files.append((f"Instrument apply (density matrix) [{prec}]", fname))

        # --- Instrument SV ---
        fig = _plot_instrument(recs, "instrument_sv", f"Instrument apply (SV) [{prec}]")
        if fig.data:
            fname = f"instrument_sv{sfx}.png"
            pio.write_image(fig, str(outdir / fname), scale=2)
            plot_files.append((f"Instrument apply (state vector) [{prec}]", fname))

    # Cross-precision comparison (if both precisions present)
    if len(by_precision) > 1:
        fig = _plot_precision_comparison(
            records, "unitary_sv",
            "Unitary apply: precision comparison (1Qop, no ensemble)",
            filter_fn=lambda r: r.get("subsystem") == "1Qop" and r.get("ensemble") == "()",
        )
        if fig.data:
            fname = "precision_unitary.png"
            pio.write_image(fig, str(outdir / fname), scale=2)
            plot_files.append(("Precision comparison: unitary", fname))

        fig = _plot_precision_comparison(
            records, "superop_dm",
            "SuperOp apply: precision comparison (1Qop, no ensemble)",
            filter_fn=lambda r: r.get("subsystem") == "1Qop" and r.get("ensemble") == "()",
        )
        if fig.data:
            fname = "precision_superop.png"
            pio.write_image(fig, str(outdir / fname), scale=2)
            plot_files.append(("Precision comparison: superop", fname))

        fig = _plot_precision_comparison(
            records, "kraus_sv",
            "Kraus trajectory: precision comparison (rank1 full, no ensemble)",
            filter_fn=lambda r: (
                r.get("source") == "rank1" and r.get("truncated") == "full" and r.get("ensemble") == "()"
            ),
        )
        if fig.data:
            fname = "precision_kraus.png"
            pio.write_image(fig, str(outdir / fname), scale=2)
            plot_files.append(("Precision comparison: Kraus trajectory", fname))

    # Write markdown report
    report = outdir / "report.md"
    with open(report, "w") as f:
        f.write("# Quax Benchmark Report\n\n")
        f.write(f"Generated from: {', '.join(json_paths)}\n\n")

        for title, fname in plot_files:
            f.write(f"## {title}\n\n")
            f.write(f"![{title}]({fname})\n\n")

        # Summary tables
        f.write("---\n\n## Summary Tables\n\n")
        for cat, cat_title in [
            ("unitary_sv", "Unitary + State Vector"),
            ("superop_dm", "SuperOp + Density Matrix"),
            ("kraus_sv", "Kraus Trajectory + State Vector"),
            ("instrument_dm", "Instrument + Density Matrix"),
            ("instrument_sv", "Instrument + State Vector"),
        ]:
            f.write(f"### {cat_title}\n\n")
            f.write(_summary_table(records, cat))
            f.write("\n")

    print(f"Report written to {report}")
    print(f"Plots written to {outdir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <json_file> [<json_file> ...]", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1:])
