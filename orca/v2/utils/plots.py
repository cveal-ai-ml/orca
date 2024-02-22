"""
Purpose: Plotting Tools
Author: Charlie
"""


import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from matplotlib.animation import PillowWriter

from utils.general import create_folder
from utils.cluster.methods import get_best_k_info

from utils.cluster.measures import get_supervised_measures
from utils.cluster.measures import get_unsupervised_measures

plt.style.use("ggplot")


def save_class_images(path, all_data):

    for current_key in all_data.keys():

        path_key = os.path.join(path, str(current_key))
        path_exemplars = os.path.join(path_key, "exemplars")
        create_folder(path_exemplars, overwrite=True)

        all_samples = all_data[current_key]["samples"]
        all_labels = all_data[current_key]["preds"]

        all_exemplars = all_data[current_key]["exemplars"]

        for i, exemplar in enumerate(all_exemplars):
            name = str(i).zfill(6) + ".png"
            path_file = os.path.join(path_exemplars, name)
            shutil.copy(exemplar, path_file)

        for label in np.unique(all_labels):
            indices = np.where(label == all_labels)
            class_samples = all_samples[indices]

            path_folder = os.path.join(path_key, str(label).zfill(6))
            create_folder(path_folder, overwrite=True)

            for i, sample in enumerate(class_samples):
                name = str(i).zfill(6) + ".png"
                path_file = os.path.join(path_folder, name)
                shutil.copy(sample, path_file)


def plot_bars(path, all_results, width=0.08, fig_size=(14, 8), font_size=24):

    # Gather: Model Names

    all_names = [model_results["name"] for model_results in all_results]

    # Gather: Bar Heights

    targets = []
    for model_results in all_results:
        for current_key in model_results.keys():
            if current_key != "name":
                targets.append(current_key)

    targets = list(np.unique(targets))

    all_groups = []

    for model_results in all_results:

        group = []
        for current_key in model_results.keys():
            if current_key in targets:
                group.append(model_results[current_key])

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

    ax.set_xlabel("Models", fontsize=font_size)
    ax.set_ylabel("Performance", fontsize=font_size)

    ax.set_ylim([0, 1])

    count = len(targets) // 2
    ax.set_xticks([r + width * count for r in range(len(all_groups[0]))],
                  targets, fontsize=font_size)

    ax.legend(loc="upper right")

    fig.tight_layout()

    fig.savefig(path)
    plt.close()


def rotate_plot(i, ax):

    ax.view_init(-150, 60 + i)


def plot_3d_scatter(path, data, labels, figsize=(14, 8), fontsize=24):

    plt.style.use("default")

    fig = plt.Figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               s=45, c=labels, cmap="rainbow")

    ax.set_title("TSNE 3D", fontsize=fontsize)
    ax.set_xlabel("x0", fontsize=fontsize)
    ax.set_ylabel("x1", fontsize=fontsize)
    ax.set_zlabel("x2", fontsize=fontsize)

    fig.tight_layout()

    fps = 20
    num_frames = 500

    plots = [ax]
    ani = animation.FuncAnimation(fig, rotate_plot,
                                  frames=num_frames, fargs=(plots))

    writer = PillowWriter(fps=fps)
    ani.save(path, writer=writer)

    plt.close()


def plot_2d_scatter(path, data, labels, figsize=(14, 8), fontsize=18):

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="rainbow")

    ax.set_title("TSNE 2D", fontsize=fontsize)
    ax.set_xlabel("x0", fontsize=fontsize)
    ax.set_ylabel("x1", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(path)

    plt.close()


def plot_features(path, data, labels):

    path_features = os.path.join(path, "predictions", "features")
    create_folder(path_features, overwrite=True)

    print("\n------ Plotting Features ------\n")

    pbar = tqdm(total=2, desc="Saving")

    path_save = os.path.join(path_features, "tsne_2d.png")
    plot_2d_scatter(path_save, data[2], labels)

    pbar.update(1)

    path_save = os.path.join(path_features, "tsne_3d.gif")
    plot_3d_scatter(path_save, data[3], labels)

    pbar.update(1)

    pbar.close()


def plot_histogram(path, data, title, figsize=(14, 8), fontsize=18):

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(np.arange(data.shape[-1]), data)

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Classes", fontsize=fontsize)
    ax.set_ylabel("Confidence", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(path)

    plt.close()


def plot_confidences(path, data, all_labels, tag, num_examples=10):

    path = os.path.join(path, "predictions")

    all_dims = list(data.keys())

    print("\n------ Plotting Confidences - %s ------\n" % tag.capitalize())

    path_root = os.path.join(path, tag)

    num_labels = len(np.unique(all_labels))
    total = len(all_dims) * num_labels * num_examples

    pbar = tqdm(total=total, desc="Saving")

    for d_tag in all_dims:

        path_folder = os.path.join(path_root, str(d_tag).zfill(4))

        for label in np.unique(all_labels):

            path_class = os.path.join(path_folder, str(label).zfill(3))
            create_folder(path_class, overwrite=True)

            indices = np.where(label == all_labels)
            class_data = data[d_tag][indices]

            for i in range(num_examples):
                filename = str(i).zfill(3) + ".png"
                path_save = os.path.join(path_class, filename)
                title = "Dims = %s" % (d_tag)
                plot_histogram(path_save, class_data[i], title)
                pbar.update(1)

    pbar.close()


def plot_gray_image(path, data, title, figsize=(8, 8), fontsize=18):

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(data, cmap="gray")

    ax.grid(False)
    ax.set_title(title, fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(path)

    plt.close()


def plot_group(path, data):

    create_folder(path, overwrite=True)

    for current_key in data.keys():
        path_file = os.path.join(path, "%s.png" % current_key)
        matrix = data[current_key]
        name = "Distances (%s)" % current_key
        plot_gray_image(path_file, matrix, name)


def plot_matrices(path, all_data):

    path = os.path.join(path, "matrices")

    print("\n------ Plotting Matrices ------\n")

    for current_key in tqdm(all_data.keys(), desc="Saving"):
        data = all_data[current_key]
        path_save = os.path.join(path, current_key)
        plot_group(path_save, data)


def plot_k_results(path, data, title, figsize=(14, 8), fontsize=18):

    for measure_key in data.keys():

        path_file = os.path.join(path, "%s.png" % measure_key)

        fig, ax = plt.subplots(figsize=figsize)

        for matrix_key in data[measure_key].keys():

            y_vals = data[measure_key][matrix_key]
            x_vals = np.arange(len(y_vals)) + 2

            if matrix_key == "max":
                ax.plot(x_vals, y_vals, "-o", linewidth=2,
                        color="darkred", label=matrix_key)
            else:
                ax.plot(x_vals, y_vals, linewidth=7, label=matrix_key)

        ax.set_xlabel("K Values", fontsize=fontsize)
        ax.set_ylabel("%s" % measure_key, fontsize=fontsize)
        ax.set_title("%s (%s)" % (title, measure_key), fontsize=fontsize)
        ax.set_xticks(np.arange(len(x_vals)) + 2)
        ax.legend()

        fig.tight_layout()
        fig.savefig(path_file)

        plt.close()


def plot_k_measures(path, all_data, tag):

    print("\n------ Plotting %s Validites (Across K) ------\n" % tag)

    for current_key in tqdm(all_data.keys(), desc="Saving"):

        data = all_data[current_key]
        path_save = os.path.join(path, current_key)
        create_folder(path_save)
        plot_k_results(path_save, data, current_key)


def get_k_measures(path, all_data, all_preds=None, labels=None):

    if labels is not None:
        tag = "Supervised"
    else:
        tag = "Unsupervised"

    print("\n------ Gathering %s Validites (Across K) ------\n" % tag)

    all_results = {}
    for current_key in tqdm(all_data.keys(), desc="Gathering"):

        data = all_data[current_key]

        if labels is not None:
            results = get_supervised_measures(data, labels)
        else:
            preds = all_preds[current_key]
            results = get_unsupervised_measures(data, preds)

        all_results[current_key] = results

    return all_results


def plot_clustering(path, all_data, preds, data):

    path = os.path.join(path, "clustering")

    # Plot: Unsupervised K Validity Measures

    u_measures = get_k_measures(path, all_data, all_preds=preds)
    plot_k_measures(path, u_measures, "k_unsupervised")

    # Plot: Supervised K Validity Measures

    s_measures = get_k_measures(path, all_data, labels=data.labels)
    plot_k_measures(path, s_measures, "k_supervised")

    # Gather: Best K Model Results
    # - Using unsupervised K validity measures

    bar_results, subset_results = get_best_k_info(u_measures, all_data, data)

    print("\n------ Saving Final Results ------\n")

    # Plot: Best K Results

    path_final = os.path.join(path, "final")
    create_folder(path_final, overwrite=True)

    for current_key in tqdm(bar_results.keys(), desc="Saving"):
        path_save = os.path.join(path_final, "%s.png" % current_key)
        plot_bars(path_save, bar_results[current_key])

        path_save = os.path.join(path_final, current_key, "class_images")
        save_class_images(path_save, subset_results[current_key])
