import os
import io
import base64
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import plot

warnings.filterwarnings("ignore", category=FutureWarning)


class RBMReportGenerator:
    def __init__(self, folder_path="reports"):
        self.folder_path = folder_path
        os.makedirs(self.folder_path, exist_ok=True)
        plt.ioff()
        # Modern color palette
        self.colors = {"bg": "#f8f9fa", "card": "#ffffff", "text": "#212529", "primary": "#4361ee"}

    def _get_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _calc_sparsity(self, activations):
        """Calculates the percentage of neurons with near-zero activity."""
        return np.mean(activations < 0.05) * 100

    def generate(
        self,
        hidden_activations,
        input_data,
        filename="rbm.html",
        num_samples=8,
    ):
        num_samples = min(num_samples, len(input_data))

        mean_act = np.mean(hidden_activations, axis=0)
        sparsity = self._calc_sparsity(hidden_activations)
        dead_units = np.sum(mean_act < 0.01)

        fig_lifetime, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(len(mean_act)), mean_act, color=self.colors["primary"], alpha=0.7)
        ax.set_title("Mean Activity per Hidden Unit (Lifetime Activity)", fontsize=12)
        ax.set_xlabel("Unit Index")
        lifetime_base64 = self._get_base64(fig_lifetime)

        fig_corr, ax = plt.subplots(figsize=(6, 5))
        corr = np.corrcoef(hidden_activations.T + 1e-9)
        sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, xticklabels=False, yticklabels=False)
        ax.set_title("Hidden Unit Correlations", fontsize=12)
        corr_base64 = self._get_base64(fig_corr)

        sample_html = ""
        for i in range(num_samples):
            side = int(np.sqrt(input_data[i].size))

            fig_in = go.Figure(
                data=go.Heatmap(
                    z=input_data[i].reshape(side, side), colorscale="Viridis", showscale=False
                )
            )
            fig_in.update_layout(
                title=f"Sample {i+1}", height=250, margin=dict(t=30, b=0, l=0, r=0)
            )
            in_div = plot(fig_in, output_type="div", include_plotlyjs="cdn" if i == 0 else False)

            fig_act = go.Figure(
                data=go.Bar(y=hidden_activations[i], marker_color=self.colors["primary"])
            )
            fig_act.update_layout(title="Activations", height=250, margin=dict(t=30, b=0, l=0, r=0))
            act_div = plot(fig_act, output_type="div", include_plotlyjs=False)

            sample_html += f"""
            <div class="sample-card">
                <div class="sample-grid">
                    {in_div}
                    {act_div}
                </div>
            </div>"""

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: 'Segoe UI', system-ui, -apple-system; background: #f0f2f5; margin: 0; color: #1c1e21; }}
                .navbar {{ background: #ffffff; padding: 1rem 2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.08); display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 1000; }}
                .container {{ max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
                .stat-card {{ background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center; border-bottom: 4px solid {self.colors['primary']}; }}
                .stat-value {{ font-size: 1.8rem; font-weight: bold; color: {self.colors['primary']}; }}
                .stat-label {{ font-size: 0.85rem; color: #65676b; text-transform: uppercase; letter-spacing: 1px; }}
                .main-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
                .card {{ background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
                .sample-grid {{ display: grid; grid-template-columns: 1fr 2fr; gap: 1rem; }}
                .sample-card {{ background: white; margin-bottom: 1rem; padding: 1rem; border-radius: 12px; transition: transform 0.2s; }}
                .sample-card:hover {{ transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
                h2 {{ font-weight: 600; margin-bottom: 1.5rem; }}
                @media (max-width: 900px) {{ .main-grid {{ grid-template-columns: 1fr; }} }}
            </style>
        </head>
        <body>
            <nav class="navbar">
                <span style="font-weight: bold; font-size: 1.2rem;">RBM Insights <span style="color:{self.colors['primary']};">Engine</span></span>
                <span style="font-size: 0.9rem; color: #65676b;">{num_samples} Samples Processed</span>
            </nav>

            <div class="container">
                <div class="stats-grid">
                    <div class="stat-card"><div class="stat-value">{sparsity:.1f}%</div><div class="stat-label">Sparsity Rate</div></div>
                    <div class="stat-card"><div class="stat-value">{dead_units}</div><div class="stat-label">Dead Neurons (&lt;1%)</div></div>
                    <div class="stat-card"><div class="stat-value">{hidden_activations.shape[1]}</div><div class="stat-label">Hidden Units</div></div>
                    <div class="stat-card"><div class="stat-value">{np.max(hidden_activations):.2f}</div><div class="stat-label">Peak Activation</div></div>
                </div>

                <div class="main-grid">
                    <div class="card">
                        <h2>Lifetime Activity Map</h2>
                        <img src="data:image/png;base64,{lifetime_base64}" style="width:100%;">
                    </div>
                    <div class="card">
                        <h2>Feature Redundancy</h2>
                        <img src="data:image/png;base64,{corr_base64}" style="width:100%;">
                    </div>
                </div>

                <h2>Individual Sample Inspection</h2>
                {sample_html}
            </div>
        </body>
        </html>
        """

        output_path = os.path.join(self.folder_path, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"🚀 Insight Report generated at: {output_path}")
