import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import os

st.set_page_config(page_title="Grokking & Model Collapse Dashboard", layout="wide")
st.title("Grokking vs Model Collapse Dashboard")

@st.cache_data
def load_data(results_dir="results/phase_transitions"):
    p = Path(results_dir)
    data = []
    if not p.exists():
        return pd.DataFrame(data)
    for cond_dir in p.iterdir():
        if not cond_dir.is_dir(): continue
        for seed_dir in cond_dir.iterdir():
            if not seed_dir.is_dir(): continue
            res_file = seed_dir / "results.json"
            if res_file.exists():
                try:
                    with open(res_file, 'r') as f:
                        res = json.load(f)
                    cfg = res['config']
                    data.append({
                        'collapse_level': cfg.get('collapse_level', 0),
                        'collapse_severity': cfg.get('collapse_severity', 0),
                        'label_noise': cfg.get('noise_fraction', 0),
                        'weight_decay': cfg.get('weight_decay', 1.0),
                        'seed': cfg.get('seed', 42),
                        'grokking_step': res.get('grokking_step', 0) if res.get('grokked', False) else 50000,
                        'final_test_acc': res.get('final_test_acc', 0),
                        'grokked': res.get('grokked', False),
                        'path': str(res_file)
                    })
                except Exception:
                    pass
    return pd.DataFrame(data)

df = load_data()

if df.empty:
    st.warning("No data found in results/phase_transitions. Run the experiment grid first.")
else:
    st.sidebar.header("Phase Diagram Explorer")
    wd = st.sidebar.select_slider("Weight Decay", options=sorted(df['weight_decay'].unique()))
    sev = st.sidebar.select_slider("Collapse Severity", options=sorted(df['collapse_severity'].unique()))

    subset = df[(df['weight_decay'] == wd) & (df['collapse_severity'] == sev)]

    if not subset.empty:
        st.subheader("Phase Diagram: Grokking Probability")

        # Aggregate across seeds
        agg = subset.groupby(['label_noise', 'collapse_level'])['grokked'].mean().reset_index()

        pivot = agg.pivot(index='label_noise', columns='collapse_level', values='grokked')
        fig = px.imshow(pivot, labels=dict(x="Collapse Level", y="Label Noise", color="Grokking Rate"),
                        title=f"Grokking Rate (wd={wd}, sev={sev})")
        st.plotly_chart(fig)

        st.subheader("Select a specific run for Training Curves")
        noise_sel = st.selectbox("Label Noise", sorted(subset['label_noise'].unique()))
        level_sel = st.selectbox("Collapse Level", sorted(subset['collapse_level'].unique()))
        seed_sel = st.selectbox("Seed", sorted(subset['seed'].unique()))

        run = subset[(subset['label_noise'] == noise_sel) &
                     (subset['collapse_level'] == level_sel) &
                     (subset['seed'] == seed_sel)]

        if not run.empty:
            run_path = run.iloc[0]['path']
            with open(run_path, 'r') as f:
                run_data = json.load(f)
            history = pd.DataFrame(run_data['history'])

            if not history.empty:
                st.subheader("Training Curves")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=history['step'], y=history['train_loss'], name='Train Loss'))
                fig2.add_trace(go.Scatter(x=history['step'], y=history['test_loss'], name='Test Loss'))
                st.plotly_chart(fig2)

                st.subheader("Accuracy & Grokking Step")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=history['step'], y=history['train_acc'], name='Train Acc'))
                fig3.add_trace(go.Scatter(x=history['step'], y=history['test_acc'], name='Test Acc'))
                if run_data.get('grokked'):
                    fig3.add_vline(x=run_data['grokking_step'], line_dash="dash", line_color="red", annotation_text="Grokking")
                st.plotly_chart(fig3)

                st.subheader("Mechanistic Timeline")
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=history['step'], y=history['weight_norm'], name='Weight Norm'))
                fig4.add_trace(go.Scatter(x=history['step'], y=history['fourier_concentration'], name='Fourier Conc'))
                st.plotly_chart(fig4)

                st.subheader("Fourier Spectrum Evolution Animation")
                # Simulate a fourier spectrum evolution for the animation
                # Since we don't have the full checkpoint files in the dashboard, we will animate a simulated spectrum
                # that peaks as fourier_concentration goes up.
                if 'fourier_concentration' in history.columns:
                    # Create simulated animation data
                    n_freqs = 59
                    frames = []

                    for idx, row in history.iterrows():
                        step = row['step']
                        conc = row['fourier_concentration']

                        # Generate spectrum
                        # Base noise
                        spectrum = np.random.uniform(0, 0.1, n_freqs)

                        # Peak around freq 1, 2, 3 as concentration increases
                        if conc > 0:
                            spectrum[1] += conc * 0.8
                            spectrum[2] += conc * 0.5
                            spectrum[3] += conc * 0.3

                        # Normalize a bit
                        spectrum = spectrum / (spectrum.sum() + 1e-10)

                        for f_idx, val in enumerate(spectrum):
                            frames.append({
                                'step': step,
                                'frequency': f_idx,
                                'amplitude': val
                            })

                    anim_df = pd.DataFrame(frames)

                    if not anim_df.empty:
                        fig5 = px.bar(anim_df, x="frequency", y="amplitude", animation_frame="step",
                                      range_y=[0, 1.0], title="Simulated Fourier Spectrum Evolution")
                        st.plotly_chart(fig5)
