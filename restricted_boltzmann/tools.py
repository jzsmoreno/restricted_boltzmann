import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from plotly.offline import plot

# Ignore all FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)


def generate_html_report(
    hidden_activations: np.ndarray,
    input_data: np.ndarray,
    output_filename: str = "report.html",
    num_samples: int = 5,
    folder_path: str = ".",
):
    """Generate an HTML report with visualizations of hidden layer activations."""

    # Ensure the folder path exists
    os.makedirs(folder_path, exist_ok=True)

    # Turn off interactive mode
    plt.ioff()

    def save_figure(fig, filename, folder_path=folder_path):
        fig.savefig(os.path.join(folder_path, filename))
        plt.close(fig)

    # Heatmap using Seaborn
    heatmap_fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(hidden_activations[:num_samples], cmap="viridis", ax=ax)
    ax.set_title("Heatmap of Hidden Layer Activations (First {} Samples)".format(num_samples))
    heatmap_filename = "heatmap.png"
    save_figure(heatmap_fig, heatmap_filename)

    # Histogram using Seaborn
    histogram_fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(hidden_activations.flatten(), bins=20, kde=True, color="skyblue", ax=ax)
    ax.set_title("Histogram of Hidden Unit Activations")
    histogram_filename = "histogram.png"
    save_figure(histogram_fig, histogram_filename)

    # Interactive plots using Plotly
    activation_vs_input_plots = []
    for i in range(num_samples):
        input_sample = input_data[i]
        sample_size = len(input_sample)

        # Calculate the closest square root dimensions
        side_length = int(np.sqrt(sample_size))
        if side_length * side_length != sample_size:
            raise ValueError(
                f"Input pattern {i+1} does not have a perfect square size. Cannot reshape to 2D array."
            )

        fig = px.imshow(
            input_sample.reshape((side_length, side_length)),
            color_continuous_scale="gray",
            title=f"Input Pattern {i+1}",
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig_div = plot(fig, output_type="div")

        bar_fig = px.bar(
            x=range(hidden_activations.shape[1]),
            y=hidden_activations[i],
            title=f"Hidden Layer Activations for Input Pattern {i+1}",
        )
        bar_fig_div = plot(bar_fig, output_type="div")

        activation_vs_input_plots.append((fig_div, bar_fig_div))

    with open(
        os.path.join(folder_path, output_filename), "w", encoding="utf-8"
    ) as f:  # Specify UTF-8 encoding
        f.write("<html>\n")
        f.write("<head><title>RBM Hidden Layer Activations Report</title></head>\n")
        f.write("<body>\n")
        f.write("<h1>RBM Hidden Layer Activations Report</h1>\n")

        f.write("<h2>Heatmap of Hidden Layer Activations</h2>\n")
        f.write('<img src="{}" alt="Heatmap">\n'.format(heatmap_filename))

        f.write("<h2>Histogram of Hidden Unit Activations</h2>\n")
        f.write('<img src="{}" alt="Histogram">\n'.format(histogram_filename))

        for i in range(num_samples):
            f.write("<h2>Activation vs Input Pattern {}</h2>\n".format(i + 1))
            f.write("<p><strong>Input Pattern:</strong></p>")
            f.write(activation_vs_input_plots[i][0])  # Plotly HTML div for input image
            f.write("<br>")
            f.write(activation_vs_input_plots[i][1])  # Plotly HTML div for activations

        f.write("</body>\n")
        f.write("</html>\n")

    plt.ion()  # Turn on interactive mode if needed
