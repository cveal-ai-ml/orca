"""
Author: Charlie
Purpose: Scoring tools
"""

import os
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.discovery.extraction import get_pca_features, get_tsne_features

plt.style.use("ggplot")


def plot_recons(path, origs, recons):

    fig, ax = plt.subplots()

    orig_grid = torchvision.utils.make_grid(origs[0], nrow=8)
    recon_grid = torchvision.utils.make_grid(recons[0], nrow=8)

    orig_grid = orig_grid.permute((1, 2, 0))
    recon_grid = recon_grid.permute((1, 2, 0))

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(orig_grid)
    ax[1].imshow(recon_grid)

    ax[0].grid(False)
    ax[1].grid(False)

    fig.tight_layout()

    fig.savefig(path)


def plot_3d(path, x, y, z, colors, title,
            fig_size=(15, 15), font_size=14):

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x, y, z, c=colors)

    ax.set_xlabel("x0", fontsize=font_size)
    ax.set_ylabel("x1", fontsize=font_size)
    ax.set_zlabel("x2", fontsize=font_size)
    ax.set_title(title, fontsize=font_size)

    fig.tight_layout()
    fig.savefig(path)


def plot_features(path, features, labels):

    all_orig = features["original"]
    all_recon = features["recon"]
    all_high = features["high"]
    all_low = features["low"]

    # Save: Low Features

    path_save = os.path.join(path, "spatial.png")
    x, y, z = all_low[:, 0], all_low[:, 1], all_low[:, 2]
    plot_3d(path_save, x, y, z, labels, "Middle Space Features")

    # Save: High Features

    path_save = os.path.join(path, "pca.png")
    reduced = get_pca_features(all_high, num_features=3)
    x, y, z = reduced[:, 0], reduced[:, 1], reduced[:, 2]
    plot_3d(path_save, x, y, z, labels, "High Dimensional Features (PCA)")

    path_save = os.path.join(path, "tsne.png")
    reduced = get_tsne_features(all_high, num_features=3)
    x, y, z = reduced[:, 0], reduced[:, 1], reduced[:, 2]
    plot_3d(path_save, x, y, z, labels, "High Dimensional Features (TSNE)")

    # Save: Reconstructed Features

    path_save = os.path.join(path, "recons.png")
    plot_recons(path_save, all_orig, all_recon)


def plot_scores(path, all_results, width=0.08,
                fig_size=(14, 8), font_size=14):
    """
    Plot confusion matrix analytics

    Parameters:
    - path (str): filesystem location to save plot
    - all_results (list[any]): all scores for each model
    - width (float): bar plot bar width
    - fig_size (tuple[int]): figure size width and height
    - font_size (int): size of the figure font
    """

    path = os.path.join(path, "scores.png")

    # Gather: Model Names

    all_names = [model_results["name"] for model_results in all_results]

    # Gather: Bar Heights

    targets = []
    all_groups = []
    for model_results in all_results:
        group = []
        for current_key in model_results.keys():
            if "score" in current_key:
                group.append(model_results[current_key])
                if current_key not in targets:
                    targets.append(current_key)

        all_groups.append(group)

    # Gather: Bar Positions

    all_positions = []

    for i in range(len(all_groups)):

        if i == 0:
            position = np.arange(len(all_groups[0]))
        else:
            position = [ele + width for ele in all_positions[i - 1]]

        all_positions.append(position)

    # Plot: Bars

    fig, ax = plt.subplots(figsize=fig_size)

    color_range = np.arange(len(all_groups)) + 1
    colors = plt.cm.turbo(color_range / float(max(color_range)))
    for i, (position, group) in enumerate(zip(all_positions, all_groups)):
        ax.bar(position, group, width=width,
               edgecolor="black", label=all_names[i], color=colors[i])

    ax.set_xlabel("Measures", fontsize=font_size)
    ax.set_ylabel("Performance", fontsize=font_size)

    ax.set_ylim([-0.05, 1.05])
    ax.set_xticks([r + width * 2 for r in range(len(all_groups[0]))],
                  targets, rotation=0, fontsize=font_size)

    ax.legend(loc="upper right")

    fig.tight_layout()

    fig.savefig(path)
    plt.close()


def plot_matrix(path, matrix, title, fig_size=(15, 15), font_size=14):
    """
    Plot and saves a comparison matrix

    Parameters:
    - path (str): filesystem location to save plot
    - matrix (np.ndarray[float]): matrix to visualize
    - title (str): title of figure
    - fig_size (tuple[int]): figure size width and height
    - font_size (int): size of the figure font
    """

    fig, ax = plt.subplots(figsize=fig_size)

    # Display: Matrix

    im = ax.imshow(matrix, cmap="gray")

    # Define: Axis Info

    ax.set_title(title, fontsize=font_size)
    ax.set_xticks(np.arange(len(matrix)))
    ax.set_yticks(np.arange(len(matrix)))

    # Define: Colorbar

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Finalize: Matrix

    ax.grid(False)
    fig.tight_layout()

    # Save: Matrix

    fig.savefig(path)
    plt.close()
