from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

if TYPE_CHECKING:
    from dara.result import RefinementResult


def visualize(
    result: RefinementResult,
    diff_offset: bool = False,
    missing_peaks: list[list[float]] | np.ndarray | None = None,
    extra_peaks: list[list[float]] | np.ndarray | None = None,
    intensity_mismatch_peaks: list[list[float]] | np.ndarray | None = None,
):
    """Visualize the result from the refinement. It uses plotly as the backend engine.

    Args:
        intensity_mismatch_peaks: optional (N, 2) array of [2theta, 0.0] markers
            for matched peaks whose calculated height/area deviates from
            observed; drawn the same way as ``missing_peaks``/``extra_peaks``
            when provided.
    """
    colormap = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    plot_data = result.plot_data

    # Create a Plotly figure with size 800x600
    fig = go.Figure()

    # Adding scatter plot for observed data
    fig.add_trace(
        go.Scatter(
            x=plot_data.x,
            y=plot_data.y_obs,
            mode="markers",
            marker=dict(color="blue", size=3, symbol="cross-thin-open"),
            name="Observed",
        )
    )

    # Adding line plot for calculated data
    fig.add_trace(
        go.Scatter(
            x=plot_data.x,
            y=plot_data.y_calc,
            mode="lines",
            line=dict(color="green", width=2),
            name="Calculated",
        )
    )

    # Adding line plot for background
    fig.add_trace(
        go.Scatter(
            x=plot_data.x,
            y=plot_data.y_bkg,
            mode="lines",
            line=dict(color="#FF7F7F", width=2),
            name="Background",
            opacity=0.5,
        )
    )

    diff = np.array(plot_data.y_obs) - np.array(plot_data.y_calc)
    diff_offset_val = 1.1 * max(diff) if diff_offset else 0  # 10 percent below

    # Adding line plot for difference
    fig.add_trace(
        go.Scatter(
            x=plot_data.x,
            y=diff - diff_offset_val,
            mode="lines",
            line=dict(color="#808080", width=1),
            name="Difference",
            hoverinfo="skip",  # "skip" to not show hover info for this trace
            opacity=0.7,
        )
    )

    # if there is no phase weight, it will return an empty dictionary (not shown in the legend)
    try:
        weight_fractions = result.get_phase_weights()
    except TypeError:
        weight_fractions = {}

    peak_data = result.peak_data
    max_y = max(np.array(result.plot_data.y_obs) + np.array(result.plot_data.y_bkg))
    min_y_diff = min(
        np.array(result.plot_data.y_obs) - np.array(result.plot_data.y_calc)
    )
    # Adding dashed lines for phases
    for i, (phase_name, phase) in enumerate(plot_data.structs.items()):
        # add area under the curve between the curve and the plot_data["y_bkg"]
        if i >= len(colormap) - 1:
            i = i % (len(colormap) - 1)

        name = (
            f"{phase_name} ({weight_fractions[phase_name] * 100:.2f} %)"
            if len(weight_fractions) > 1
            else phase_name
        )
        fig.add_trace(
            go.Scatter(
                x=plot_data.x,
                y=plot_data.y_bkg,
                mode="lines",
                line=dict(color=colormap[i], width=0),
                fill=None,
                showlegend=False,
                hoverinfo="none",
                legendgroup=phase_name,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_data.x,
                y=np.array(phase) + np.array(plot_data.y_bkg),
                mode="lines",
                line=dict(color=colormap[i], width=1.5),
                fill="tonexty",
                name=name,
                visible="legendonly",
                legendgroup=phase_name,
            )
        )
        refl = peak_data[peak_data["phase"] == phase_name]["2theta"]
        intensity = peak_data[peak_data["phase"] == phase_name]["intensity"]
        fig.add_trace(
            go.Scatter(
                x=refl,
                y=np.ones(len(refl)) * (i + 1) * -max_y * 0.1 + min_y_diff,
                mode="markers",
                marker={
                    "symbol": 142,
                    "size": 5,
                    "color": colormap[i],
                },
                name=name,
                legendgroup=phase_name,
                showlegend=False,
                visible="legendonly",
                text=[f"{x:.2f}, {y:.2f}" for x, y in zip(refl, intensity)],
                hovertemplate="%{text}",
            )
        )

    if missing_peaks is not None:
        missing_peaks = np.array(missing_peaks).reshape(-1, 2)
        fig.add_trace(
            go.Scatter(
                x=missing_peaks[:, 0],
                y=np.zeros_like(missing_peaks[:, 0]),
                mode="markers",
                marker=dict(color="#f9726a", symbol=53, size=10, opacity=0.8),
                name="Missing peaks",
                showlegend=True,
                visible="legendonly",
                text=[f"{x:.2f}, {y:.2f}" for x, y in missing_peaks],
            )
        )

    if extra_peaks is not None:
        extra_peaks = np.array(extra_peaks).reshape(-1, 2)
        fig.add_trace(
            go.Scatter(
                x=extra_peaks[:, 0],
                y=np.zeros_like(extra_peaks[:, 0]),
                mode="markers",
                marker=dict(color="#335da0", symbol=53, size=10, opacity=0.8),
                name="Extra peaks",
                showlegend=True,
                visible="legendonly",
                text=[f"{x:.2f}, {y:.2f}" for x, y in extra_peaks],
                hovertemplate="%{text}",
            )
        )

    if intensity_mismatch_peaks is not None:
        intensity_mismatch_peaks = np.array(intensity_mismatch_peaks).reshape(-1, 2)
        fig.add_trace(
            go.Scatter(
                x=intensity_mismatch_peaks[:, 0],
                y=np.zeros_like(intensity_mismatch_peaks[:, 0]),
                mode="markers",
                marker=dict(color="#FFD700", symbol=53, size=10, opacity=0.8),
                name="Intensity mismatch",
                showlegend=True,
                visible="legendonly",
                text=[f"{x:.2f}, {y:.2f}" for x, y in intensity_mismatch_peaks],
                hovertemplate="%{text}",
            )
        )

    title = f"{result.lst_data.pattern_name} (Rwp={result.lst_data.rwp:.2f}%)"

    # Updating layout with titles and labels
    fig.update_layout(
        autosize=True,
        xaxis=dict(
            range=[min(plot_data.x), max(plot_data.x)],
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        ),
        title=title,
        xaxis_title="2θ [°]",
        yaxis_title="Intensity",
        legend_title="",
        font=dict(family="Arial, sans-serif", color="RebeccaPurple"),
        plot_bgcolor="white",
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
        legend_tracegroupgap=1,
    )

    fig.add_hline(y=0, line_width=1)

    # add tick
    fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor="black", ticklen=10)
    fig.update_yaxes(ticks="outside", tickwidth=1, tickcolor="black", ticklen=10)

    return fig
