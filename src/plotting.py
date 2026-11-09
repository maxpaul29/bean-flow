import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


# common log-scale color limits for several matrices, so their heatmaps are comparable
# a log scale is needed because sigmoid gradients are orders of magnitude smaller than relu gradients
def shared_limits(matrices):
    values = np.concatenate([m.ravel() for m in matrices])
    values = values[values > 0]
    return values.min(), values.max()


# draw one heatmap on the given axis: x = training step, y = layer (layer 1 at the top), color = gradient norm
def plot_heatmap(grad_matrix, ax, title, vmin, vmax, record_every=10):
    num_records, num_layers = grad_matrix.shape
    image = ax.imshow(
        grad_matrix.T,
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=[0, num_records * record_every, num_layers + 0.5, 0.5],
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer (1 = input side, 8 = output)")
    ax.set_yticks(range(1, num_layers + 1))
    return image


# one heatmap in its own figure with a colorbar and a one-sentence caption
def save_heatmap(grad_matrix, title, caption, path, vmin=None, vmax=None, record_every=10):
    if vmin is None or vmax is None:
        vmin, vmax = shared_limits([grad_matrix])

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    image = plot_heatmap(grad_matrix, ax, title, vmin, vmax, record_every)
    fig.colorbar(image, ax=ax, label="Gradient norm (log scale)")
    fig.supxlabel(caption, fontsize=9, style="italic")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# several heatmaps side by side on one shared color scale, one colorbar for all
# used for sigmoid vs relu, clipped vs unclipped and seed 42 vs the second seed
def plot_side_by_side(matrices, panel_titles, title, caption, path, record_every=10):
    vmin, vmax = shared_limits(matrices)

    fig, axes = plt.subplots(1, len(matrices), figsize=(5.5 * len(matrices), 4.5),
                             sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, matrix, panel_title in zip(axes, matrices, panel_titles):
        image = plot_heatmap(matrix, ax, panel_title, vmin, vmax, record_every)
    fig.colorbar(image, ax=axes, label="Gradient norm (log scale)")
    fig.suptitle(title)
    fig.supxlabel(caption, fontsize=9, style="italic")
    fig.savefig(path, dpi=150)
    plt.close(fig)
