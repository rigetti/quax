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
    python benchmarks/analyze.py benchmarks/results/results.json

Produces:
    benchmarks/results/report.md   — Markdown with embedded plot images
    benchmarks/results/*.png       — Individual plot images
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Rigetti plot template
# ---------------------------------------------------------------------------

rigetti_template = go.layout.Template()
rigetti_template.layout = go.Layout(
    colorway=["#00b5ad", "#ef476f", "#ffc504", "#3c47d9", "#8a8b92", "#0d0d36"],
)
rigetti_template.data.scatter = [go.Scatter(marker={"size": 10, "line": {"width": 2, "color": "DarkSlateGrey"}})]
pio.templates["rigetti"] = rigetti_template
pio.templates.default = "ggplot2+rigetti"

# Plot dimensions (~30% smaller in width, taller for readability)
_WIDTH = 700
_HEIGHT = 525

# ---------------------------------------------------------------------------
# Dimension string -> qubit count
# ---------------------------------------------------------------------------

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

_RE_UNITARY = re.compile(r"test_targeted_apply_unitary\[(?P<sys>\w+)-ens(?P<ens>\([^)]*\))-(?P<sub>\w+)\]")
_RE_SUPEROP = re.compile(r"test_targeted_apply_superop\[(?P<sys>\w+)-ens(?P<ens>\([^)]*\))-(?P<sub>\w+)\]")
_RE_KRAUS = re.compile(
    r"test_kraus_trajectory\[(?P<sys>\w+)-ens(?P<ens>\([^)]*\))"
    r"-(?P<src>rank\d+|depolarizing)-(?P<trunc>full|trunc)-(?P<sub>\w+)\]"
)
_RE_INST_DM = re.compile(r"test_instrument_dm\[(?P<sys>\w+)-ens(?P<ens>\([^)]*\))-(?P<itype>ideal|noisy)\]")
_RE_INST_SV = re.compile(r"test_instrument_sv\[(?P<sys>\w+)-ens(?P<ens>\([^)]*\))-(?P<itype>ideal|noisy)\]")


def _classify(bench: dict) -> dict | None:
    """Parse a benchmark dict and return a record with extracted fields, or None."""
    name = bench["name"]
    mean_ms = bench["stats"]["mean"] * 1e3  # seconds -> ms
    median_ms = bench["stats"]["median"] * 1e3
    stddev_ms = bench["stats"]["stddev"] * 1e3

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
                "mean_ms": mean_ms,
                "median_ms": median_ms,
                "stddev_ms": stddev_ms,
                "name": name,
            }
            if "ens" in m.groupdict():
                rec["ensemble"] = m.group("ens")
            if "sub" in m.groupdict():
                rec["subsystem"] = m.group("sub")
            if category == "kraus_sv":
                rec["source"] = m.group("src")
                rec["truncated"] = m.group("trunc")
            if category in ("instrument_dm", "instrument_sv"):
                rec["inst_type"] = m.group("itype")
            return rec
    return None


# ---------------------------------------------------------------------------
# Load + merge JSON results
# ---------------------------------------------------------------------------


def _load_results(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for bench in data.get("benchmarks", []):
            rec = _classify(bench)
            if rec:
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------


def _hardware_summary() -> str:
    """Return a short markdown block describing the machine."""
    lines = []
    lines.append(f"- **Date**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- **OS**: {platform.system()} {platform.release()}")
    lines.append(f"- **CPU**: {platform.processor() or platform.machine()}")
    cpu_count = os.cpu_count()
    if cpu_count:
        lines.append(f"- **CPU cores**: {cpu_count}")
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    lines.append(f"- **RAM**: {kb / 1024 / 1024:.0f} GB")
                    break
    except OSError:
        pass
    lines.append(f"- **Python**: {platform.python_version()}")
    try:
        import jax

        lines.append(f"- **JAX**: {jax.__version__}")
        lines.append(f"- **JAX platforms**: {', '.join(str(d) for d in jax.devices())}")
    except ImportError:
        pass
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Consistent color per op size across all plots
_OP_COLORS: dict[str, str] = {
    "1Qop": "#00b5ad",
    "2Qop": "#ef476f",
    "3Qop": "#ffc504",
    "4Qop": "#3c47d9",
}


def _op_color(sub: str) -> str:
    return _OP_COLORS.get(sub, "#8a8b92")


def _base_layout(title: str, yaxis_title: str = "Mean time (ms)") -> dict:
    return {
        "title": title,
        "xaxis_title": "Number of qubits",
        "yaxis_title": yaxis_title,
        "yaxis_type": "log",
        "width": _WIDTH,
        "height": _HEIGHT,
        "legend": {"font": {"size": 10}},
    }


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
            y=[r["mean_ms"] for r in recs],
            error_y={"type": "data", "array": [r["stddev_ms"] for r in recs], "visible": True},
            mode="markers+lines",
            marker={"color": color, "symbol": symbol},
            line={"color": color, "dash": dash, "width": 2},
            name=label,
        )
    )


def _save(fig: go.Figure, path: Path) -> None:
    pio.write_image(fig, str(path), scale=3)


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------


def _plot_apply_by_ensemble(
    records: list[dict],
    category: str,
    ensemble: str,
    title: str,
) -> go.Figure:
    """One line per op size for a fixed ensemble.  Color = op size."""
    fig = go.Figure()
    cat_recs = [r for r in records if r["category"] == category and r.get("ensemble") == ensemble]
    if not cat_recs:
        return fig

    subsystems = sorted({r["subsystem"] for r in cat_recs})
    for sub in subsystems:
        subset = [r for r in cat_recs if r["subsystem"] == sub]
        _add_line(fig, subset, sub, _op_color(sub))

    fig.update_layout(**_base_layout(title))
    return fig


def _plot_kraus_by_ensemble(
    records: list[dict],
    ensemble: str,
    title: str,
) -> go.Figure:
    """One line per (op_size, source, trunc) for a fixed ensemble.  Color = op size."""
    fig = go.Figure()
    cat_recs = [r for r in records if r["category"] == "kraus_sv" and r.get("ensemble") == ensemble]
    if not cat_recs:
        return fig

    combos = sorted({(r["subsystem"], r["source"], r["truncated"]) for r in cat_recs})
    for sub, src, trunc in combos:
        subset = [r for r in cat_recs if r["subsystem"] == sub and r["source"] == src and r["truncated"] == trunc]
        dash = "dash" if trunc == "trunc" else "solid"
        label = f"{sub} {src} ({trunc})"
        _add_line(fig, subset, label, _op_color(sub), dash=dash)

    fig.update_layout(**_base_layout(title))
    return fig


def _plot_kraus_ensemble_scaling(records: list[dict], title: str) -> go.Figure:
    """X = ensemble size, Y = mean time averaged over systems/sources/trunc.  Color = op size."""
    fig = go.Figure()
    cat_recs = [r for r in records if r["category"] == "kraus_sv"]
    if not cat_recs:
        return fig

    # Parse ensemble tuple to a scalar size (e.g. "(16,)" -> 16, "()" -> 1)
    def _ens_size(ens_str: str) -> int:
        inner = ens_str.strip("()")
        if not inner or inner == "":
            return 1
        return int(inner.rstrip(","))

    subsystems = sorted({r["subsystem"] for r in cat_recs})
    for sub in subsystems:
        sub_recs = [r for r in cat_recs if r["subsystem"] == sub]
        # Group by ensemble, average over all other variables
        ens_values = sorted({r["ensemble"] for r in sub_recs}, key=_ens_size)
        x_vals = []
        y_vals = []
        for ens in ens_values:
            group = [r for r in sub_recs if r["ensemble"] == ens]
            avg_ms = sum(r["mean_ms"] for r in group) / len(group)
            x_vals.append(_ens_size(ens))
            y_vals.append(avg_ms)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+lines",
                marker={"color": _op_color(sub)},
                line={"color": _op_color(sub), "width": 2},
                name=sub,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Ensemble size",
        yaxis_title="Mean time (ms) — averaged over systems & sources",
        yaxis_type="log",
        xaxis_type="log",
        width=_WIDTH,
        height=_HEIGHT,
        legend={"font": {"size": 10}},
    )
    return fig


def _plot_instrument(records: list[dict], category: str, title: str) -> go.Figure:
    """Instrument plot: one line per (ensemble, inst_type)."""
    fig = go.Figure()
    cat_recs = [r for r in records if r["category"] == category]
    if not cat_recs:
        return fig

    ensembles = sorted({r["ensemble"] for r in cat_recs})
    colors = ["#00b5ad", "#ef476f", "#ffc504", "#3c47d9", "#8a8b92", "#0d0d36"]
    for i, ens in enumerate(ensembles):
        c = colors[i % len(colors)]
        for itype in ["ideal", "noisy"]:
            subset = [r for r in cat_recs if r["ensemble"] == ens and r["inst_type"] == itype]
            dash = "dash" if itype == "noisy" else "solid"
            symbol = "x" if itype == "noisy" else "circle"
            label = f"ens={ens}  {itype}"
            _add_line(fig, subset, label, c, dash=dash, symbol=symbol)

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

    rows.sort(key=lambda r: (r["num_qubits"], r.get("ensemble", ""), r.get("subsystem", "")))

    lines = ["| System | Qubits | Ensemble | Op | Mean (ms) | Std (ms) |"]
    lines.append("|--------|--------|----------|----|-----------|----------|")
    for r in rows:
        ens = r.get("ensemble", "-")
        sub = r.get("subsystem", "-")
        lines.append(
            f"| {r['system']} | {r['num_qubits']} | {ens} | {sub} | {r['mean_ms']:.2f} | {r['stddev_ms']:.2f} |"
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

    # Collect all ensembles present
    ensembles = sorted({r.get("ensemble", "()") for r in records})

    plot_files: list[tuple[str, str]] = []  # (title, relative_path)

    # --- Unitary: one plot per ensemble, color = op size ---
    for ens in ensembles:
        title = f"Unitary apply (SV) — ens={ens}"
        fig = _plot_apply_by_ensemble(records, "unitary_sv", ens, title)
        if fig.data:
            ens_tag = ens.replace("(", "").replace(")", "").replace(",", "") or "scalar"
            fname = f"unitary_sv_ens{ens_tag}.png"
            _save(fig, outdir / fname)
            plot_files.append((title, fname))

    # --- SuperOp: one plot per ensemble, color = op size ---
    for ens in ensembles:
        title = f"SuperOp apply (DM) — ens={ens}"
        fig = _plot_apply_by_ensemble(records, "superop_dm", ens, title)
        if fig.data:
            ens_tag = ens.replace("(", "").replace(")", "").replace(",", "") or "scalar"
            fname = f"superop_dm_ens{ens_tag}.png"
            _save(fig, outdir / fname)
            plot_files.append((title, fname))

    # --- Kraus: one plot per ensemble, color = op size ---
    for ens in ensembles:
        title = f"Kraus trajectory (SV) — ens={ens}"
        fig = _plot_kraus_by_ensemble(records, ens, title)
        if fig.data:
            ens_tag = ens.replace("(", "").replace(")", "").replace(",", "") or "scalar"
            fname = f"kraus_sv_ens{ens_tag}.png"
            _save(fig, outdir / fname)
            plot_files.append((title, fname))

    # --- Kraus: ensemble scaling (averaged) ---
    fig = _plot_kraus_ensemble_scaling(records, "Kraus trajectory: ensemble scaling")
    if fig.data:
        fname = "kraus_ensemble_scaling.png"
        _save(fig, outdir / fname)
        plot_files.append(("Kraus trajectory: ensemble scaling", fname))

    # --- Instrument DM ---
    fig = _plot_instrument(records, "instrument_dm", "Instrument apply (DM)")
    if fig.data:
        fname = "instrument_dm.png"
        _save(fig, outdir / fname)
        plot_files.append(("Instrument apply (density matrix)", fname))

    # --- Instrument SV ---
    fig = _plot_instrument(records, "instrument_sv", "Instrument apply (SV)")
    if fig.data:
        fname = "instrument_sv.png"
        _save(fig, outdir / fname)
        plot_files.append(("Instrument apply (state vector)", fname))

    # Write markdown report
    report = outdir / "report.md"
    with open(report, "w") as f:
        f.write("# Quax Benchmark Report\n\n")
        f.write("## Environment\n\n")
        f.write(_hardware_summary())
        f.write("\n\n")
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
