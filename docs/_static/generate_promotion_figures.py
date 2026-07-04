"""Generate figures for the promotion documentation.

Run from the docs/_static directory:
    JAX_ENABLE_X64=1 python generate_promotion_figures.py

64-bit precision is enforced so that any future additions get full precision
by default.  The off-diagonal values in the coherent-extension Weyl-Liouville
figure (~0.06, ~0.02) are physical, not numerical noise, and are identical at
32-bit and 64-bit.
"""

import sys

sys.path.insert(0, "../../src")

import jax

# Must be set before any JAX computation.
jax.config.update("jax_enable_x64", True)

import plotly.graph_objects as go

import quax as qx


def _contiguous_groups(positions: list[int]) -> list[tuple[int, int]]:
    """Return (start, end) pairs for each run of consecutive integers."""
    if not positions:
        return []
    groups, start, end = [], positions[0], positions[0]
    for p in positions[1:]:
        if p == end + 1:
            end = p
        else:
            groups.append((start, end))
            start = end = p
    groups.append((start, end))
    return groups


def _band_trace(
    x0: float, x1: float, y0: float, y1: float, hatch: str, group: str, name: str, show: bool
) -> go.Scatter:
    return go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        fill="toself",
        fillpattern=dict(shape=hatch, fgcolor="rgba(0,0,0,0.30)", size=6),
        fillcolor="rgba(0,0,0,0)",
        line=dict(width=0),
        mode="lines",
        legendgroup=group,
        name=name,
        showlegend=show,
    )


def _use_index_axes(fig: go.Figure, n: int) -> None:
    """Use numeric cell coordinates while preserving existing axis labels."""
    for trace in fig.data:
        x_labels = getattr(trace, "x", None)
        y_labels = getattr(trace, "y", None)
        if x_labels is None or y_labels is None:
            continue
        if len(x_labels) != n or len(y_labels) != n:
            continue

        x_labels = list(x_labels)
        y_labels = list(y_labels)
        indices = list(range(n))
        trace.x = indices
        trace.y = indices
        fig.update_xaxes(tickvals=indices, ticktext=x_labels)
        fig.update_yaxes(tickvals=indices, ticktext=y_labels)
        return


def _apply_subspace_hatching(
    fig: go.Figure,
    n: int,
    cross: list[int],
    compl: list[int],
) -> None:
    """Overlay hatched bands on cross-subspace and complement rows/columns.

    Uses "x" hatching for cross-subspace indices and "+" for complement.
    Bands are added to the first subplot (row=1, col=1).
    """
    _use_index_axes(fig, n)

    for indices, hatch, group, label in [
        (cross, "x", "cross", "cross-subspace"),
        (compl, "+", "compl", "complement"),
    ]:
        first = True
        for start, end in _contiguous_groups(indices):
            fig.add_trace(
                _band_trace(start - 0.5, end + 0.5, -0.5, n - 0.5, hatch, group, label, show=first), row=1, col=1
            )
            fig.add_trace(
                _band_trace(-0.5, n - 0.5, start - 0.5, end + 0.5, hatch, group, label, show=False), row=1, col=1
            )
            first = False

    fig.update_xaxes(range=[-0.5, n - 0.5], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-0.5, n - 0.5], autorange="reversed", showgrid=False, zeroline=False)
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", x=0.0, y=-0.12, xanchor="left", yanchor="top", font=dict(size=11)),
        margin=dict(b=70),
    )


def _add_subspace_labels(fig: go.Figure, d_comp: int, D: int) -> None:
    """Add hatched overlays labelling cross-subspace and complement rows/columns.

    Rows/columns index density-matrix elements |i><j| in the computational
    basis.  "x" hatching marks cross-subspace elements (exactly one of i, j
    in the computational subspace); "+" marks complement elements (both outside).
    """
    n = D * D
    cross = [k for k in range(n) if (k // D < d_comp) != (k % D < d_comp)]
    compl = [k for k in range(n) if k // D >= d_comp and k % D >= d_comp]
    _apply_subspace_hatching(fig, n, cross, compl)


def _add_weyl_subspace_labels(fig: go.Figure, d_comp: int, D: int) -> None:
    """Add hatched overlays labelling cross-subspace and complement Weyl operators.

    Rows/columns index Weyl-Heisenberg operators W_{x,z}.  An operator is
    "computational" when both x and z are in {0,...,d_comp-1}, "complement"
    when both are outside, and "cross-subspace" otherwise.  "x" hatching marks
    cross-subspace operators; "+" marks complement operators.
    """
    from quax._operator_basis import _xz_pairs

    xz_list = _xz_pairs(D)
    n = len(xz_list)  # D^2

    cross = [k for k, (x, z) in enumerate(xz_list) if (x < d_comp) != (z < d_comp)]
    compl = [k for k, (x, z) in enumerate(xz_list) if x >= d_comp and z >= d_comp]
    _apply_subspace_hatching(fig, n, cross, compl)


def _make_weyl_figures(p: float = 0.1) -> None:
    """Weyl-Liouville transfer-matrix figures for the coherent/incoherent extensions."""
    kraus = qx.to_kraus(qx.depolarizing_channel_superoperator(p, (2,)))
    promoted_coherent = qx.promote(kraus, (3,))
    promoted_incoherent = qx.promote_incoherent(kraus, (3,))

    D = promoted_coherent.dims[0][0]
    range_color = (-1.0, 1.0)

    fig_c = qx.plot(promoted_coherent, range_color=range_color)
    _add_weyl_subspace_labels(fig_c, d_comp=2, D=D)
    fig_c.update_layout(title=dict(text=f"Coherent extension (p={p})", x=0.5))
    fig_c.write_image("promotion-weyl-coherent.png", scale=3)
    print("Generated: promotion-weyl-coherent.png")

    fig_i = qx.plot(promoted_incoherent, range_color=range_color)
    _add_weyl_subspace_labels(fig_i, d_comp=2, D=D)
    fig_i.update_layout(title=dict(text=f"Incoherent extension (p={p})", x=0.5))
    fig_i.write_image("promotion-weyl-incoherent.png", scale=3)
    print("Generated: promotion-weyl-incoherent.png")


def _make_block_figures(p: float = 0.1) -> None:
    """Computational-basis superoperator figures showing cross-subspace coherence."""
    kraus = qx.to_kraus(qx.depolarizing_channel_superoperator(p, (2,)))
    superop_c = qx.to_superop(qx.promote(kraus, (3,)))
    superop_i = qx.to_superop(qx.promote_incoherent(kraus, (3,)))

    D = superop_c.dims[0][0]

    fig_c = qx.plot(superop_c)
    _add_subspace_labels(fig_c, d_comp=2, D=D)
    fig_c.update_layout(title=dict(text=f"Coherent extension — computational basis (p={p})", x=0.5))
    fig_c.write_image("promotion-block-coherent.png", scale=3)
    print("Generated: promotion-block-coherent.png")

    fig_i = qx.plot(superop_i)
    _add_subspace_labels(fig_i, d_comp=2, D=D)
    fig_i.update_layout(title=dict(text=f"Incoherent extension — computational basis (p={p})", x=0.5))
    fig_i.write_image("promotion-block-incoherent.png", scale=3)
    print("Generated: promotion-block-incoherent.png")


if __name__ == "__main__":
    _make_weyl_figures()
    _make_block_figures()
    print("\nAll promotion figures generated successfully!")
