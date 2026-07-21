import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
import torch.nn as nn
import os
import sys
from typing import Dict, List, Any

# Add the project root to the sys path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from viz.attention_evolution import load_attention_weights
from viz.circuit_analysis import track_circuit_formation
from viz.weight_analysis import extract_weight_norm_trajectory, plot_loss_landscape_contour, compute_hessian_max_eigenvalue
from src.model import ModularArithmeticTransformer

# Note: In a real environment, we'd read this from a config or directory structure.
# For testing, we point it to the dummy checkpoint.
AVAILABLE_RUNS = {
    "dummy_run": ["tests/data/dummy_checkpoint.pt"]
}

app = dash.Dash(__name__, title="Grokking Mechanisms Dashboard")

app.layout = html.Div([
    html.H1("Grokking Mechanisms Dashboard"),

    html.Div([
        html.Label("Select Run:"),
        dcc.Dropdown(
            id='run-selector',
            options=[{'label': k, 'value': k} for k in AVAILABLE_RUNS.keys()],
            value='dummy_run'
        )
    ], style={'width': '30%', 'margin-bottom': '20px'}),

    dcc.Tabs([
        dcc.Tab(label='Loss & Accuracy', children=[
            html.Div([
                html.Label("These metrics are usually extracted from training logs or final test evaluations."),
                dcc.Graph(id='loss-acc-graph'),
            ])
        ]),

        dcc.Tab(label='Circuit Formation', children=[
            html.Div([
                dcc.Graph(id='induction-score-graph'),
                html.P("Induction scores track when heads begin copying from previous tokens.")
            ])
        ]),

        dcc.Tab(label='Weight Space Analysis', children=[
            html.Div([
                dcc.Graph(id='weight-norm-graph'),
                html.P("Weight norm trajectories often indicate grokking (L2 norm reduction).")
            ]),
            html.Div([
                html.H3("Loss Landscape (at final checkpoint)"),
                dcc.Graph(id='loss-landscape-graph')
            ])
        ]),

        dcc.Tab(label='Attention Patterns', children=[
            html.Div([
                html.Label("Select Checkpoint Step:"),
                dcc.Slider(
                    id='step-slider',
                    min=0,
                    max=1,
                    step=1,
                    marks={0: 'Init'},
                    value=0
                ),
                dcc.Graph(id='attention-heatmap-graph')
            ])
        ])
    ])
])

@app.callback(
    Output('loss-acc-graph', 'figure'),
    Input('run-selector', 'value')
)
def update_loss_acc_graph(run_name):
    # Dummy mock for loss/accuracy curve since actual training logs are not hooked up
    fig = go.Figure()
    steps = np.linspace(0, 1000, 20)
    fig.add_trace(go.Scatter(x=steps, y=np.exp(-steps/200), name='Train Loss', mode='lines'))
    fig.add_trace(go.Scatter(x=steps, y=1 - np.exp(-steps/300), name='Test Acc', mode='lines', yaxis='y2'))

    fig.update_layout(
        title='Mock Loss and Accuracy Curves',
        xaxis_title='Steps',
        yaxis_title='Loss',
        yaxis2=dict(
            title='Accuracy',
            overlaying='y',
            side='right',
            range=[0, 1]
        )
    )
    return fig

@app.callback(
    Output('induction-score-graph', 'figure'),
    Input('run-selector', 'value')
)
def update_circuit_graph(run_name):
    paths = AVAILABLE_RUNS.get(run_name, [])
    if not paths:
        return go.Figure()

    steps = list(range(len(paths)))
    metrics = track_circuit_formation(paths, steps)

    if not metrics:
        return go.Figure()

    fig = go.Figure()
    scores = metrics['induction_scores'] # (steps, heads)

    for h in range(scores.shape[1]):
        fig.add_trace(go.Scatter(
            x=steps, y=scores[:, h],
            mode='lines+markers',
            name=f'Head {h+1}'
        ))

    fig.update_layout(
        title='Induction Score vs Step',
        xaxis_title='Step (Index)',
        yaxis_title='Induction Score'
    )
    return fig

@app.callback(
    Output('weight-norm-graph', 'figure'),
    Input('run-selector', 'value')
)
def update_weight_norm_graph(run_name):
    paths = AVAILABLE_RUNS.get(run_name, [])
    if not paths:
        return go.Figure()

    steps = list(range(len(paths)))
    metrics = extract_weight_norm_trajectory(paths, steps)

    if not metrics:
        return go.Figure()

    fig = go.Figure()

    for k, v in metrics.items():
        if k == 'steps': continue
        fig.add_trace(go.Scatter(
            x=steps, y=v,
            mode='lines+markers',
            name=k
        ))

    fig.update_layout(
        title='Weight Norms vs Step',
        xaxis_title='Step (Index)',
        yaxis_title='L2 Norm'
    )
    return fig

@app.callback(
    Output('loss-landscape-graph', 'figure'),
    Input('run-selector', 'value')
)
def update_loss_landscape(run_name):
    paths = AVAILABLE_RUNS.get(run_name, [])
    if not paths:
        return go.Figure()

    # Get last checkpoint
    path = paths[-1]

    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        config = checkpoint.get('config', {})
        model = ModularArithmeticTransformer(
            prime=config.get('prime', 59),
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            d_ff=config.get('d_ff', 512),
            n_layers=config.get('n_layers', 1)
        )
        model.load_state_dict(checkpoint['model_state'])

        # Dummy data
        x = torch.randint(0, config.get('prime', 59), (16, 2))
        y = (x[:, 0] + x[:, 1]) % config.get('prime', 59)
        loss_fn = nn.CrossEntropyLoss()

        # Fast grid size for dashboard responsiveness
        loss_grid = plot_loss_landscape_contour(model, loss_fn, (x, y), grid_size=5)

        scale = 1.0
        alphas = np.linspace(-scale, scale, 5)
        betas = np.linspace(-scale, scale, 5)

        # Log scale
        Z = np.log1p(loss_grid.T)

        fig = go.Figure(data=go.Contour(
            z=Z,
            x=alphas,
            y=betas,
            colorscale='Viridis'
        ))

        fig.update_layout(
            title='Loss Landscape (Filter Normalized Directions)',
            xaxis_title='Direction 1',
            yaxis_title='Direction 2'
        )
        return fig
    except Exception as e:
        print(f"Error generating landscape: {e}")
        return go.Figure()

@app.callback(
    [Output('step-slider', 'max'),
     Output('step-slider', 'marks')],
    Input('run-selector', 'value')
)
def update_slider(run_name):
    paths = AVAILABLE_RUNS.get(run_name, [])
    max_val = max(0, len(paths) - 1)
    marks = {i: str(i) for i in range(len(paths))}
    return max_val, marks

@app.callback(
    Output('attention-heatmap-graph', 'figure'),
    [Input('run-selector', 'value'),
     Input('step-slider', 'value')]
)
def update_attention_heatmap(run_name, step_idx):
    paths = AVAILABLE_RUNS.get(run_name, [])
    if not paths or step_idx >= len(paths):
        return go.Figure()

    path = paths[step_idx]

    try:
        attn = load_attention_weights(path)
        # We'll plot Head 1 for simplicity in the Plotly graph
        h = 0
        heatmap_data = attn[h].numpy()

        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Key", y="Query", color="Attention"),
            x=['a', 'b'],
            y=['a', 'b'],
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            title=f'Attention Head {h+1} Pattern at Step {step_idx}'
        )
        return fig
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return go.Figure()

if __name__ == '__main__':
    # Print a warning/notice since we're just verifying the script works
    print("Starting dashboard... (Press Ctrl+C to quit)")
    # Using a random port to avoid conflicts
    app.run_server(debug=False, port=8050)