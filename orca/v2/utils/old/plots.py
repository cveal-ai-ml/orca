"""
Author: Charlie
Purpose: Scoring tools
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def plot_k_clustering(path, data, title, fig_size=(14, 8), font_size=14):

    group = []
    for current_key in data.keys():
        if "fuse" in current_key:
            group.append(current_key)
        elif "preds" in current_key:
            continue
        else:
            fig, ax = plt.subplots(figsize=fig_size)

            num_classes = len(data[current_key])
            ax.plot(np.arange(num_classes) + 3,
                    data[current_key], "-o", color="darkgreen")

            ax.set_xlabel("Number of Clusters", fontsize=font_size)
            ax.set_ylabel(current_key)

            if current_key == "rand":
                ax.set_ylim([-0.05, 1.05])

            fig.tight_layout()
            fig.savefig(os.path.join(path, "%s.png" % current_key))

    if len(group) > 0:

        fig, ax = plt.subplots(figsize=fig_size)
        for current_key in group:
            num_classes = len(data[current_key])
            ax.plot(np.arange(num_classes) + 3, data[current_key],
                    "-o", label="%s" % current_key)

        ax.set_xlabel("Number of Clusters", fontsize=font_size)
        ax.set_ylabel("Fused Measures")
        ax.set_ylim([-0.05, 1.05])
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(path, "fused.png"))

    plt.close("all")


def plot_histogram(path, data, title, fig_size=(14, 8), font_size=14):

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(data)
    ax.set_title(title, fontsize=font_size)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_features(path, data, labels, title, fig_size=(14, 8), font_size=14):

    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="rainbow")
    ax.set_title(title, fontsize=font_size)

    fig.tight_layout()
    fig.savefig(path)
    plt.close()


def plot_matrix(path, data, title, fig_size=(14, 8), font_size=14):

    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(data, cmap="gray")

    ax.set_title(title, fontsize=font_size)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars(path, data, width=0.01, fig_size=(14, 8), font_size=14):

    fig, ax = plt.subplots(figsize=fig_size)

    ax.bar(np.arange(len(data)), data, width=width,
           edgecolor="black", color="darkblue")

    tag = int(path.split("/")[-1].strip(".png").strip("class_"))
    title = "Class Confidences - Truth Label Class %s" % tag

    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel("Classes", fontsize=font_size)
    ax.set_ylabel("Frequency", fontsize=font_size)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
