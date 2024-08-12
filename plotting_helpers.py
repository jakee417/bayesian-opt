from IPython.display import display, Markdown
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure
from typing import Dict, List, Optional, Any
from numpy.typing import ArrayLike
from ax.service.ax_client import AxClient
from ax.core.types import TModelPredict
import plotly.graph_objects as go
import numpy as np


def print_markdown(markdown: str) -> None:
    display(Markdown(markdown))


def render_plotly_html(fig: go.Figure) -> None:
    fig.show()
    display(
        Markdown(
            fig.to_html(
                include_plotlyjs="cdn",
            )
        )
    )


uncertainty_kwargs = dict(fill="toself", hoverinfo="skip", showlegend=True, opacity=0.6)


def get_line_kwargs(color: Optional[str] = None) -> Dict[str, Any]:
    marker = dict(color=color) if color else dict()
    return dict(mode="lines", marker=marker, hoverinfo="skip", showlegend=True)


def get_scatter_kwargs(color: Optional[str] = None) -> Dict[str, Any]:
    marker = dict(color=color) if color else dict(size=10)
    return dict(marker=marker, mode="markers", showlegend=True, opacity=1.0)


def plot_groundtruth(x: ArrayLike, y: ArrayLike, kwargs: Dict[str, Any]) -> go.Scatter:
    return go.Scatter(x=x, y=y, name="f", **kwargs)


# Plotting Helper Functions
def plot_observed(
    ax_client: AxClient,
    gold_fn,
    i: int,
    kwargs,
    visible: bool = False,
) -> go.Scatter:
    _x = ax_client.get_trial_parameters(i).get("x", 0.0)
    _y = gold_fn(_x)
    return go.Scatter(visible=visible, x=[_x], y=[_y], **kwargs, name=f"i={i + 1}")


def plot_predictions(
    x: np.ndarray, predictions: Dict[int, TModelPredict], i: int
) -> go.Scatter:
    # Predictions for gold content.
    y_hat = np.array(predictions[i][0]["gold"])
    # The double "gold" key is the AxSweep way to index into Covariance matrix.
    y_hat_cov = np.array(predictions[i][1]["gold"]["gold"])
    y_upper = y_hat + y_hat_cov
    y_lower = y_hat - y_hat_cov
    return go.Scatter(
        visible=False,
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        **uncertainty_kwargs,
        name=f"p={i + 1}",
    )


def plot_acquisition(
    x: ArrayLike,
    acquisitions: Dict[int, List[List[float]]],
    i: int,
    kwargs,
) -> List[go.Scatter]:
    kwargs = kwargs.copy()
    plots = []
    for idx, j in enumerate(acquisitions[i]):
        if idx == 1:
            kwargs["showlegend"] = False
        plots.append(
            go.Scatter(
                visible=False,
                x=x,
                y=j,
                **kwargs,
                name=f"a={i + 1}",
            )
        )
    return plots


def plot_surrogate_and_acquisition(
    x: ArrayLike,
    y: ArrayLike,
    gold_fn,
    ax_client: AxClient,
    predictions: Dict[int, TModelPredict],
    acquisitions: Dict[int, List[List[float]]],
    line_kwargs: Dict[str, Any],
) -> Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_titles=["Surrogate", "Acquisition"]
    )
    fig.add_trace(plot_groundtruth(x=x, y=y, kwargs=get_line_kwargs()), row=1, col=1)
    n = len(ax_client.generation_strategy._generator_runs)
    idx = []
    for i in range(n):
        if i in acquisitions:
            idx.append(len(fig.data))
            [
                fig.add_trace(j, row=2, col=1)
                for j in plot_acquisition(
                    x=x, acquisitions=acquisitions, i=i, kwargs=line_kwargs
                )
            ]

        if i in predictions:
            fig.add_trace(plot_predictions(x=x, predictions=predictions, i=i))
        fig.add_trace(
            plot_observed(
                ax_client=ax_client,
                gold_fn=gold_fn,
                i=i,
                kwargs=get_scatter_kwargs(),
                # We want to always show the first Sobol points.
                visible=len(idx) == 0,
            )
        )

    # Setup animation
    visible = [False] * len(fig.data)
    for i in range(idx[0]):
        visible[i] = True
    steps = [
        dict(
            method="update",
            label="i = 0",
            args=[{"visible": visible.copy(), "title": "0"}],
        )
    ]
    PLOTS = idx[-1] - idx[-2]
    for i in idx:
        for j in range(i, i + PLOTS):
            visible[j] = True
        steps.append(
            dict(
                method="update",
                label=f"i = {i // PLOTS}",
                args=[{"visible": visible.copy(), "title": str(i)}],
            )
        )
        for j in range(i, i + PLOTS - 1):
            visible[j] = False

    fig.update_xaxes(range=[-1, 7])
    fig.update_layout(
        title="Gold Search",
        sliders=[dict(active=0, steps=steps)],
    )
    return fig
