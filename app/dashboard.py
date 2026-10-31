"""Dash front end for the scoring API.

The dashboard holds no model and no decision logic: it collects an application,
asks the API, and renders the answer, exactly as a production front end would.

Usage:
    uvicorn app.api:app --port 8000     # first terminal
    python app/dashboard.py             # second terminal
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dcc, html

API_URL = os.environ.get("CREDIT_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 10

APPROVE_COLOR = "#1baf7a"
REJECT_COLOR = "#eb6834"

# Field name as the API expects it, label, default value and input step.
FIELDS = [
    ("RevolvingUtilizationOfUnsecuredLines", "Revolving utilisation", 0.36, 0.01),
    ("age", "Age", 55, 1),
    ("MonthlyIncome", "Monthly income", 3325, 100),
    ("DebtRatio", "Debt ratio", 0.65, 0.01),
    ("NumberOfOpenCreditLinesAndLoans", "Open credit lines", 12, 1),
    ("NumberRealEstateLoansOrLines", "Real estate loans", 1, 1),
    ("NumberOfTime30-59DaysPastDueNotWorse", "Late 30-59 days", 0, 1),
    ("NumberOfTime60-89DaysPastDueNotWorse", "Late 60-89 days", 0, 1),
    ("NumberOfTimes90DaysLate", "Late 90+ days", 0, 1),
    ("NumberOfDependents", "Dependents", 0, 1),
]


def field_input(name: str, label: str, default: float, step: float) -> html.Div:
    """Return one labelled numeric input."""
    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.85rem"}),
            dcc.Input(id=name, type="number", value=default, step=step, style={"width": "100%"}),
        ],
        style={"marginBottom": "0.75rem"},
    )


def reason_figure(reasons: list[dict]) -> go.Figure:
    """Return a bar chart of the contributions that drove a decline."""
    figure = go.Figure(
        go.Bar(
            x=[reason["contribution"] for reason in reasons],
            y=[reason["feature"] for reason in reasons],
            orientation="h",
            marker_color=REJECT_COLOR,
        )
    )
    figure.update_layout(
        title="Why this application was declined",
        xaxis_title="Contribution to risk (log odds)",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        height=260,
    )
    return figure


def render(result: dict) -> list:
    """Turn an API response into the result panel."""
    rejected = result["decision"] == "reject"
    color = REJECT_COLOR if rejected else APPROVE_COLOR

    panel = [
        html.H3(
            "Declined" if rejected else "Approved",
            style={"color": color, "marginBottom": "0.5rem"},
        ),
        html.P(f"Default probability: {result['probability']:.2%}"),
        html.P(f"Credit score: {result['score']}"),
        html.P(f"Decision cut-off: {result['threshold']:.2%}", style={"color": "#898781"}),
    ]
    if rejected:
        panel.append(dcc.Graph(figure=reason_figure(result["reason_codes"])))
    return panel


app = Dash(__name__, title="Credit Risk Scoring")

app.layout = html.Div(
    [
        html.H2("Credit risk scoring"),
        html.P(f"Scored by the API at {API_URL}", style={"color": "#898781"}),
        html.Div(
            [
                html.Div(
                    [field_input(*field) for field in FIELDS],
                    style={"width": "320px", "marginRight": "3rem"},
                ),
                html.Div(id="result", style={"flex": "1"}),
            ],
            style={"display": "flex"},
        ),
        html.Button("Score application", id="submit", n_clicks=0),
    ],
    style={"maxWidth": "900px", "margin": "2rem auto", "fontFamily": "system-ui, sans-serif"},
)


@app.callback(
    Output("result", "children"),
    Input("submit", "n_clicks"),
    [State(name, "value") for name, *_ in FIELDS],
    prevent_initial_call=True,
)
def score(_: int, *values: float) -> list:
    """Send the form to the API and render whatever comes back."""
    payload = {name: value for (name, *_), value in zip(FIELDS, values, strict=True)}

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        return [
            html.P(f"The API at {API_URL} is not responding. Start it with 'uvicorn app.api:app'.")
        ]

    # A 422 means the schema rejected the application before the model saw it.
    if response.status_code != 200:
        return [html.P(f"The API rejected this application ({response.status_code}).")]

    return render(response.json())


if __name__ == "__main__":
    app.run(debug=False)
