from functools import singledispatch
from itertools import product
from typing import List, Optional, Tuple

import jax.numpy as jnp

try:
    import plotly.colors as pc
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.graph_objs import Figure
except ImportError as e:
    raise ImportError("plotly is required for visualization. Install it with: pip install rigetti-quax[plot]") from e

from ._operator_basis import _xz_pairs
from ._quantum_objects import (
    DensityMatrix,
    Operator,
    PauliLiouville,
    QuantumObject,
    StateVector,
    SuperOperator,
)
from ._superoperator_transformations import to_pauli_liouville


def _computational_basis_labels(dims: Tuple[int, ...]) -> List[str]:
    """Generate computational basis labels for a multi-qudit system.

    :param dims: Per-qudit dimensions, e.g. ``(2, 2)`` for two qubits.
    :returns: Labels such as ``["|00⟩", "|01⟩", "|10⟩", "|11⟩"]``.
    """
    indices = product(*(range(d) for d in dims))
    return ["|" + "".join(str(i) for i in idx) + "⟩" for idx in indices]


def _weyl_labels(dims: Tuple[int, ...]) -> List[str]:
    """Generate axis labels for a multi-qudit Weyl-Liouville matrix.

    For d=2 subsystems, uses Pauli names (I, X, Y, Z).
    For d>2, uses Wxz notation (W00, W01, ...).
    Multi-qudit labels are tensor products joined by empty string.
    """
    pauli_labels = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}
    single_labels = []
    for d in dims:
        pairs = _xz_pairs(d)
        if d == 2:
            single_labels.append([pauli_labels[p] for p in pairs])
        else:
            single_labels.append([f"W{x}{z}" for x, z in pairs])

    return ["".join(combo) for combo in product(*single_labels)]


# ---------------------------------------------------------------------------
# Low-level plot helpers
# ---------------------------------------------------------------------------


def plot_pauli_transfer_matrix(
    pauli_liouville: PauliLiouville,
    title: str = "Pauli Transfer Matrix",
    range_color: Optional[Tuple[float, float]] = None,
) -> Figure:
    """
    Plot a Pauli-Liouville (d=2) or Weyl-Liouville (d>2) transfer matrix.

    For d=2, plots the real-valued Pauli transfer matrix with Pauli labels.
    For d>2, plots the magnitude of the complex Weyl-Liouville matrix with
    Weyl labels (W00, W01, ...).

    :param pauli_liouville: A PauliLiouville quantum object.
    :param title: Title for the figure.
    :param range_color: Color range for the colorbar. If ``None``, adapts to the data.
    """
    colorscale = [(0.0, "#ef476f"), (0.5, "#f8f8f8"), (1.0, "#00b5ad")]

    matrix = pauli_liouville.matrix
    labels = _weyl_labels(pauli_liouville.dims[0])

    plot_data = jnp.real(matrix)
    if range_color is None:
        max_val = float(jnp.max(jnp.abs(plot_data)))
        range_color = (-max_val, max_val)
    rounded = jnp.around(plot_data, decimals=2)
    mask = jnp.abs(rounded) < 1e-4
    text = [
        ["" if bool(mask[i, j]) else f"{float(rounded[i, j]):.2f}" for j in range(rounded.shape[1])]
        for i in range(rounded.shape[0])
    ]

    fig = px.imshow(
        plot_data,
        x=labels,
        y=labels,
        labels=dict(x="Input Operator", y="Output Operator"),
        color_continuous_scale=colorscale,
        width=575,
        height=575,
        range_color=range_color,
        text_auto=False,
    )
    fig.update_traces(text=text, texttemplate="%{text}")

    fig.update_layout(
        coloraxis=dict(
            showscale=True,
            colorbar=dict(thickness=10, len=0.5, ypad=0),
        ),
        font=dict(
            size=12,
        ),
        title=f"{title}",
        xaxis=dict(
            ticks="inside",
            tickson="boundaries",
        ),
        yaxis=dict(
            ticks="inside",
            tickson="boundaries",
            autorange="reversed",
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Single-dispatch ``plot`` function
# ---------------------------------------------------------------------------


@singledispatch
def plot(obj: QuantumObject, **kwargs) -> Figure:  # type: ignore[type-arg]
    """Plot a quantum object.

    This is a single-dispatch function that selects the appropriate
    visualization based on the input type:

    * :class:`StateVector` – polar bar chart of computational-basis amplitudes
      coloured by imaginariness and patterned by sign.
    * :class:`DensityMatrix` – heatmap of matrix elements (real part).
    * :class:`Operator` / :class:`SuperOperator` – Pauli transfer matrix
      (converted via :func:`to_pauli_liouville`).
    * :class:`QuantumInstrument` – subplot grid of per-outcome Pauli transfer
      matrices.

    :param obj: Any supported quantum object.
    :param kwargs: Forwarded to the type-specific plotting function.
    :returns: A Plotly ``Figure``.
    :raises NotImplementedError: If the type has no registered plot.
    """
    raise NotImplementedError(f"plot is not implemented for type {type(obj).__name__}")


@plot.register(StateVector)
def _plot_state_vector(
    state_vector: StateVector,
    title: str = "State Vector",
    area_proportional: bool = False,
) -> Figure:
    """
    Polar bar chart of state-vector amplitudes.

    The angular coordinate corresponds to each computational-basis state.
    Bar height (radial extent) represents the amplitude magnitude.  Colour
    encodes ideality: teal for purely real, magenta for purely
    imaginary, and light grey for equally mixed.  A hatched pattern
    distinguishes negative amplitudes (Re < 0) from positive ones (solid).

    :param state_vector: A :class:`StateVector` (no ensemble dimensions).
    :param title: Figure title.
    :param area_proportional: If ``True``, the area of each annular sector is
        proportional to the amplitude magnitude rather than the radial extent.
    """
    if state_vector.ensemble_size != ():
        raise ValueError("plot(StateVector) does not support ensemble dimensions")
    colourscale = [
        [0.0, "#00b5ad"],
        [0.5, "#f8f8f8"],
        [1.0, "#ef476f"],
    ]
    amplitudes = jnp.array(state_vector.data).flatten()
    magnitudes = jnp.abs(amplitudes)
    phases = jnp.angle(amplitudes)
    # im/|α| ∈ [-1, 1]: -1 → red, 0 → grey, +1 → teal; map to [0, 1] for colorscale
    im_ratio = jnp.where(magnitudes > 0, jnp.imag(amplitudes) / magnitudes, 0.0)
    color_norm = ((im_ratio + 1) / 2).tolist()
    labels = _computational_basis_labels(state_vector.dims)
    colors = pc.sample_colorscale(colourscale, color_norm)
    patterns = ["" if jnp.cos(ph) >= 0 else "/" for ph in phases]
    n = len(labels)
    width = 360.0 / n  # in degrees
    thetas = [i * width for i in range(n)]

    # Hollow centre: offset the radial axis so bars start away from the origin
    max_mag = float(jnp.max(magnitudes)) if float(jnp.max(magnitudes)) > 0 else 1.0
    hole = max_mag * 0.3

    if area_proportional:
        # Area of annular sector: A = ½ Δθ (r_outer² − r_inner²)
        # Solve for r_outer so that A ∝ magnitude:
        #   r_outer = sqrt(r_inner² + 2 * mag * scale / Δθ)
        # where scale normalises the largest bar to the same max as the linear case.
        delta_theta = jnp.radians(width)
        max_r_linear = max_mag + hole
        # scale factor so max-magnitude bar has same r_outer as linear case
        scale = delta_theta * (max_r_linear**2 - hole**2) / (2 * max_mag) if max_mag > 0 else 1.0
        r_values = jnp.sqrt(hole**2 + 2 * magnitudes * scale / delta_theta)
    else:
        r_values = magnitudes + hole

    # Hide zero-amplitude bars by making them fully transparent
    line_widths = [1 if float(m) > 0 else 0 for m in magnitudes]
    colors = [c if float(m) > 0 else "rgba(0,0,0,0)" for c, m in zip(colors, magnitudes)]

    hover_text = [
        f"{lab}<br>|α|: {m:.4f}<br>Amplitude: {a.real:.4f}{a.imag:+.4f}i<br>Phase: {ph:.4f} rad"
        for lab, m, a, ph in zip(labels, magnitudes, amplitudes, phases)
    ]

    r_max = float(jnp.max(r_values)) + hole + 0.01

    # Legend entries for the hatching pattern and imaginariness colour
    legend_positive = go.Scatterpolar(
        r=[None],
        theta=[None],
        mode="markers",
        marker=dict(symbol="square", size=12, color="#d0d0d0", line=dict(color="#0d0d36", width=1)),
        name="Re ≥ 0 (solid)",
    )
    legend_negative = go.Scatterpolar(
        r=[None],
        theta=[None],
        mode="markers",
        marker=dict(symbol="square", size=12, color="#d0d0d0", line=dict(color="#0d0d36", width=1.5)),
        name="Re < 0 (hatched)",
    )
    legend_im_pos = go.Scatterpolar(
        r=[None],
        theta=[None],
        mode="markers",
        marker=dict(symbol="square", size=12, color="#00b5ad", line=dict(color="#0d0d36", width=1)),
        name="Im > 0",
    )
    legend_im_neg = go.Scatterpolar(
        r=[None],
        theta=[None],
        mode="markers",
        marker=dict(symbol="square", size=12, color="#ef476f", line=dict(color="#0d0d36", width=1)),
        name="Im < 0",
    )
    legend_im_zero = go.Scatterpolar(
        r=[None],
        theta=[None],
        mode="markers",
        marker=dict(symbol="square", size=12, color="#f8f8f8", line=dict(color="#0d0d36", width=1)),
        name="Im ≈ 0",
    )

    fig = go.Figure(
        data=[
            go.Barpolar(
                r=r_values,
                theta=thetas,
                width=[width] * n,
                base=[hole] * n,
                marker=dict(
                    color=colors,
                    pattern=dict(shape=patterns, fgcolor="#0d0d36", size=6),
                    line=dict(color="#0d0d36", width=line_widths),
                ),
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False,
            ),
            legend_positive,
            legend_negative,
            legend_im_pos,
            legend_im_neg,
            legend_im_zero,
        ],
    )

    fig.update_layout(
        template="plotly_white",
        title=title,
        showlegend=True,
        legend=dict(x=0.0, y=-0.05, orientation="h", font=dict(size=11)),
        polar=dict(
            angularaxis=dict(
                tickvals=thetas,
                ticktext=labels,
                direction="clockwise",
            ),
            radialaxis=dict(
                range=[0, r_max],
                visible=False,
            ),
            hole=hole / r_max,
        ),
        width=575,
        height=575,
    )
    return fig


@plot.register(DensityMatrix)
def _plot_density_matrix(
    density_matrix: DensityMatrix,
    title: str = "Density Matrix",
    range_color: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Heatmap of a density matrix where colour encodes phase and opacity encodes magnitude.

    Phase uses a cyclic colourmap (teal → yellow → magenta → blue → teal).
    Magnitude is mapped to opacity so that larger elements are more vivid.
    Text colour adapts for readability against the effective background.

    :param density_matrix: A :class:`DensityMatrix` (no ensemble dimensions).
    :param title: Figure title.
    :param range_color: Symmetric normalisation range. If ``None``, adapts to the data.
    """
    if density_matrix.ensemble_size != ():
        raise ValueError("plot(DensityMatrix) does not support ensemble dimensions")

    cyclic_colorscale = [
        "#e2d9e2",
        "#99e1de",
        "#76ACB4",
        "#53778A",
        "#304260",
        "#48375A",
        "#83617E",
        "#BE8BA1",
        "#F9B5C5",
        "#e2d9e2",
    ]

    matrix = density_matrix.matrix
    abs_mat = jnp.abs(matrix)
    abs_max = float(jnp.max(abs_mat))
    if abs_max == 0:
        abs_max = 1.0
    if range_color is None:
        range_color = (-abs_max, abs_max)

    labels = _computational_basis_labels(density_matrix.dims)
    n = len(labels)

    # Phase → colour via cyclic colorscale
    phases = jnp.angle(matrix)
    phase_norm = jnp.clip(((phases + jnp.pi) / (2 * jnp.pi)).ravel(), 0, 1)
    phase_colors = pc.sample_colorscale(cyclic_colorscale, phase_norm.tolist(), colortype="rgb")

    # Magnitude → opacity
    alphas = jnp.clip(abs_mat / abs_max, 0, 1)

    # Parse RGB values and compute perceptual brightness for text contrast
    rgb_array = jnp.array([[int(x) for x in str(s)[4:-1].split(",")] for s in phase_colors]).reshape(n, n, 3)
    brightness = jnp.sum(rgb_array / 255.0 * jnp.array([0.2126, 0.7152, 0.0722]), axis=-1) * alphas + (1 - alphas)

    # Adaptive text threshold and pre-computed rounded values
    text_threshold = float(jnp.mean(abs_mat) - jnp.std(abs_mat))
    real_rounded = jnp.around(jnp.real(matrix), decimals=2)
    imag_rounded = jnp.around(jnp.imag(matrix), decimals=2)

    # Build shapes (cell rectangles) and annotations (text labels)
    shapes = [
        dict(
            type="rect",
            x0=j - 0.5,
            y0=i - 0.5,
            x1=j + 0.5,
            y1=i + 0.5,
            fillcolor=f"rgba({int(rgb_array[i, j, 0])}, {int(rgb_array[i, j, 1])}, "
            f"{int(rgb_array[i, j, 2])}, {float(alphas[i, j]):.3f})",
            line=dict(width=0.5, color="#f8f8f8"),
        )
        for i in range(n)
        for j in range(n)
    ]

    annotations = []
    for i in range(n):
        for j in range(n):
            if float(abs_mat[i, j]) < text_threshold:
                continue
            re_val, im_val = float(real_rounded[i, j]), float(imag_rounded[i, j])
            re_str = f"{re_val:.2f}" if abs(re_val) >= 1e-4 else ""
            im_str = f"{im_val:+.2f}i" if abs(im_val) >= 1e-4 else ""
            text = "<br>".join(filter(None, [re_str, im_str]))
            if text:
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=text,
                        showarrow=False,
                        xref="x",
                        yref="y",
                        font=dict(size=10, color="#f8f8f8" if float(brightness[i, j]) < 0.5 else "#0d0d36"),
                    )
                )

    # Invisible scatter for the phase colourbar
    import math

    phase_colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=[[i / (len(cyclic_colorscale) - 1), c] for i, c in enumerate(cyclic_colorscale)],
            showscale=True,
            cmin=-math.pi,
            cmax=math.pi,
            color=[0],
            colorbar=dict(
                title="Phase (rad)",
                thickness=10,
                len=0.5,
                ypad=0,
                tickvals=[-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi],
                ticktext=["-π", "-π/2", "0", "π/2", "π"],
            ),
        ),
        hoverinfo="none",
        showlegend=False,
    )

    fig = go.Figure(data=[phase_colorbar_trace])
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        title=title,
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            tickvals=list(range(n)),
            ticktext=labels,
            range=[-0.5, n - 0.5],
            constrain="domain",
        ),
        yaxis=dict(
            tickvals=list(range(n)),
            ticktext=labels,
            range=[n - 0.5, -0.5],
            constrain="domain",
            scaleanchor="x",
        ),
        width=575,
        height=575,
    )
    return fig


@plot.register(Operator)
def _plot_operator(
    operator: Operator,
    title: str = "Pauli Transfer Matrix",
    range_color: Optional[Tuple[float, float]] = None,
) -> Figure:
    """
    Plot an operator as its Pauli transfer matrix.

    :param operator: Any :class:`Operator` subclass (Unitary, Observable, …).
    :param title: Figure title.
    :param range_color: Color range for the colorbar. If ``None``, adapts to the data.
    """
    if operator.ensemble_size != ():
        raise ValueError("plot(Operator) does not support ensemble dimensions")
    if (any(d > 2 for dim in operator.dims for d in dim)) and (title == "Pauli Transfer Matrix"):
        title = "Weyl-Liouville Matrix"
    pauli_liouville = to_pauli_liouville(operator)
    return plot_pauli_transfer_matrix(pauli_liouville, title=title, range_color=range_color)


@plot.register(SuperOperator)
def _plot_superoperator(
    superoperator: SuperOperator,
    title: str = "Pauli Transfer Matrix",
    range_color: Optional[Tuple[float, float]] = None,
) -> Figure:
    """
    Plot a superoperator as its Pauli transfer matrix.

    :param superoperator: Any :class:`SuperOperator` subclass (SuperOp, Choi, KrausMap, PauliLiouville, …).
    :param title: Figure title.
    :param range_color: Color range for the colorbar. If ``None``, adapts to the data.
    """
    if (any(d > 2 for dim in superoperator.dims for d in dim)) and (title == "Pauli Transfer Matrix"):
        title = "Weyl-Liouville Matrix"
    pauli_liouville = to_pauli_liouville(superoperator)
    return plot_pauli_transfer_matrix(pauli_liouville, title=title, range_color=range_color)
