import base64
import io
import os
import warnings
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from plotly.offline import plot

warnings.filterwarnings("ignore", category=FutureWarning)


class RBMReportGenerator:
    def __init__(self, folder_path: str = "reports") -> None:
        self.folder_path = folder_path
        os.makedirs(self.folder_path, exist_ok=True)
        plt.ioff()
        # Modern color palette
        self.colors: Dict[str, str] = {
            "bg": "#f8f9fa",
            "card": "#ffffff",
            "text": "#212529",
            "primary": "#4361ee",
        }

    def _get_base64(self, fig: plt.Figure, dpi: int = 120) -> str:
        """Converts a matplotlib figure to a base64-encoded PNG string.

        Parameters
        ----------
        fig : `plt.Figure`
            The matplotlib figure to convert.
        dpi : `int`, optional
            Resolution of the output image (default: 120).

        Returns
        -------
        base64_str : `str`
            Base64-encoded PNG representation of the figure.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, facecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _calc_sparsity(self, activations: np.ndarray, threshold: float = 1e-2) -> float:
        """Calculates the percentage of neurons with near-zero activity.

        Parameters
        ----------
        activations : `np.ndarray`
            Hidden unit activations of shape (n_samples, n_hidden_units).
        threshold : `float`, optional
            Activation values below this threshold are considered "near-zero"
            (default: 1e-2).

        Returns
        -------
        sparsity : `float`
            Percentage of activations below the threshold.
        """
        return float(np.mean(activations < threshold) * 100)

    def _validate_inputs(
        self,
        hidden_activations: np.ndarray,
        input_data: np.ndarray,
        num_samples: int,
    ) -> int:
        """Validates the inputs to the generate method.

        Parameters
        ----------
        hidden_activations : `np.ndarray`
            Hidden unit activations of shape (n_samples, n_hidden_units).
        input_data : `np.ndarray`
            Input data of shape (n_samples, n_features).
        num_samples : `int`
            Number of samples to display in the report.

        Returns
        -------
        num_samples : `int`
            The validated (clamped) number of samples to display.

        Raises
        ------
        ValueError
            If inputs are empty, have mismatched sample counts, or
            `num_samples` is not positive.
        """
        if not isinstance(hidden_activations, np.ndarray) or not isinstance(input_data, np.ndarray):
            raise ValueError("hidden_activations and input_data must be numpy arrays.")

        if hidden_activations.ndim != 2:
            raise ValueError(
                f"hidden_activations must be 2D, got shape {hidden_activations.shape}."
            )
        if input_data.ndim != 2:
            raise ValueError(f"input_data must be 2D, got shape {input_data.shape}.")

        if hidden_activations.shape[0] == 0 or input_data.shape[0] == 0:
            raise ValueError("hidden_activations and input_data must not be empty.")

        if hidden_activations.shape[0] != input_data.shape[0]:
            raise ValueError(
                "hidden_activations and input_data must have the same number of samples. "
                f"Got {hidden_activations.shape[0]} and {input_data.shape[0]}."
            )

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")

        return min(num_samples, len(input_data))

    def _pad_to_square(self, sample: np.ndarray) -> np.ndarray:
        """Pads a 1D sample to the nearest perfect square for image display.

        Parameters
        ----------
        sample : `np.ndarray`
            A 1D array of feature values.

        Returns
        -------
        padded : `np.ndarray`
            A 1D array padded with zeros to the next perfect square length.
        """
        size = sample.size
        side = int(np.ceil(np.sqrt(size)))
        padded_size = side * side
        if padded_size == size:
            return sample
        padded = np.zeros(padded_size, dtype=sample.dtype)
        padded[:size] = sample
        return padded

    def generate(
        self,
        hidden_activations: np.ndarray,
        input_data: np.ndarray,
        filename: str = "rbm.html",
        num_samples: int = 8,
    ) -> None:
        """Generates an interactive HTML report with RBM insights.

        Parameters
        ----------
        hidden_activations : `np.ndarray`
            Hidden unit activations of shape (n_samples, n_hidden_units).
        input_data : `np.ndarray`
            Input data of shape (n_samples, n_features).
        filename : `str`, optional
            Output HTML filename (default: "rbm.html").
        num_samples : `int`, optional
            Number of samples to display in the report (default: 8).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If inputs are invalid (empty, mismatched shapes, or non-positive
            `num_samples`).
        """
        num_samples = self._validate_inputs(hidden_activations, input_data, num_samples)

        # Replace NaN values in activations with 0 to avoid propagating errors
        activations_clean = np.nan_to_num(hidden_activations, nan=0.0)

        mean_act = np.mean(activations_clean, axis=0)
        sparsity = self._calc_sparsity(activations_clean)
        dead_units = int(np.sum(mean_act < 1e-2))

        fig_lifetime, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(len(mean_act)), mean_act, color=self.colors["primary"], alpha=0.7)
        ax.set_title("Mean Activity per Hidden Unit (Lifetime Activity)", fontsize=12)
        ax.set_xlabel("Unit Index")
        lifetime_base64 = self._get_base64(fig_lifetime)

        # Handle the single-sample correlation edge case
        if activations_clean.shape[0] > 1:
            fig_corr, ax = plt.subplots(figsize=(6, 5))
            corr = np.corrcoef(activations_clean.T + 1e-9)
            sns.heatmap(
                corr,
                cmap="RdBu_r",
                center=0,
                ax=ax,
                xticklabels=False,
                yticklabels=False,
            )
            ax.set_title("Hidden Unit Correlations", fontsize=12)
            corr_base64 = self._get_base64(fig_corr)
        else:
            # With a single sample, correlations are undefined; show a placeholder
            fig_corr, ax = plt.subplots(figsize=(6, 5))
            ax.text(
                0.5,
                0.5,
                "Not enough samples\nfor correlation analysis",
                ha="center",
                va="center",
                fontsize=12,
                color="#65676b",
            )
            ax.axis("off")
            corr_base64 = self._get_base64(fig_corr)

        sample_html = ""
        for i in range(num_samples):
            # Pad non-square samples to the nearest perfect square for display
            padded_sample = self._pad_to_square(input_data[i])
            side = int(np.sqrt(padded_sample.size))

            fig_in = go.Figure(
                data=go.Heatmap(
                    z=padded_sample.reshape(side, side),
                    colorscale="gray",
                    showscale=False,
                )
            )
            fig_in.update_layout(
                title=f"Sample {i+1}", height=250, margin=dict(t=30, b=0, l=0, r=0)
            )
            in_div = plot(fig_in, output_type="div", include_plotlyjs="cdn" if i == 0 else False)

            fig_act = go.Figure(
                data=go.Bar(y=activations_clean[i], marker_color=self.colors["primary"])
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
                    <div class="stat-card"><div class="stat-value">{dead_units}</div><div class="stat-label">Dead Neurons (<1%)</div></div>
                    <div class="stat-card"><div class="stat-value">{activations_clean.shape[1]}</div><div class="stat-label">Hidden Units</div></div>
                    <div class="stat-card"><div class="stat-value">{np.max(activations_clean):.2f}</div><div class="stat-label">Peak Activation</div></div>
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
        try:
            print(f"🚀 Insight Report generated at: {output_path}")
        except UnicodeEncodeError:
            # Fallback for terminals that don't support emoji (e.g., cp1252 on Windows)
            print(f"Insight Report generated at: {output_path}")
