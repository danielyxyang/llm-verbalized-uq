import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from benchmark.evaluation import aggregate_responses, calibration_curve
from utils_ext.plot import get_figlayout

logger = logging.getLogger(__name__)

def create_subplots(row_names, col_names, width=4, **args):
    fig, axes = plt.subplots(**get_figlayout(nrows=len(row_names), ncols=len(col_names), width=width), layout="constrained", squeeze=False, **args)
    for ax, row in zip(axes[:,0], row_names):
        ax.set_ylabel(row, size="large")
    for ax, col in zip(axes[0], col_names):
        ax.set_title(col)
    return fig, axes

def plot_heatmap(ax, heatmap, row_labels, col_labels, plot_mean=True, format="{:.3f}", **kwargs):
    if plot_mean:
        row_labels = row_labels + ["all"]
        col_labels = col_labels + ["all"]
        heatmap_row_mean = np.mean(heatmap, axis=0, keepdims=True)
        heatmap_column_mean = np.mean(heatmap, axis=1, keepdims=True)
        heatmap = np.block([
            [heatmap,          heatmap_column_mean],
            [heatmap_row_mean, np.full((1, 1), np.nan)],
        ])

    # plot heatmap
    ax.imshow(heatmap, **kwargs)

    # add labels
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # add heatmap annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, format.format(heatmap[i, j]), ha="center", va="center")

def plot_annotation(ax, text, **kwargs):
    ax.text(0.05, 0.95, text, va="top", transform=ax.transAxes, **kwargs)

def plot_calibration_curve(ax, y_true, y_pred, n_bins, color="tab:blue", labelfmt="{:.3f}", labelsize=None):
    prob_true, prob_pred, bins, bin_count = calibration_curve(y_true, y_pred, n_bins=n_bins)
    ece = np.sum(bin_count / np.sum(bin_count) * np.abs(prob_true - prob_pred), where=bin_count > 0)

    from matplotlib.colors import to_rgba
    bin_count_rel = bin_count / np.max(bin_count) # normalize by max
    bin_count_rel = np.log(1 + bin_count_rel) / np.log(2) # apply log scale
    bin_count_rel = 0.1 + 0.9 * np.nan_to_num(bin_count_rel) # rescale to range [0.1, 1.0]
    bin_colors = [to_rgba(color, alpha) for alpha in bin_count_rel]
    ax.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", label="perfect calibration")
    # ax.plot(prob_pred, prob_true, marker="o", color=color)
    ax.bar(bins[:-1], prob_true, 1/n_bins, align="edge", color=bin_colors, label="actual calibration")
    plot_annotation(ax, f"ECE: {labelfmt}".format(ece), fontsize=labelsize)

def plot_confidence_histogram(ax, y_pred, n_bins, normalized=True, **kwargs):
    weights = np.full(len(y_pred), 1 / len(y_pred)) if normalized and len(y_pred) > 0 else None
    ax.hist(y_pred, bins=np.linspace(0.0, 1.0, n_bins + 1), range=(0, 1), weights=weights, **kwargs)
    plot_annotation(ax, f"distinct: {len(np.unique(y_pred))}")

def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Figure saved to \"{path}\".")



def create_fig_accuracy_distribution(name, y_true_all, names, row_key, col_key, n_bins, save=None):
    fig, ax = create_subplots(names[row_key] + ["all"], names[col_key] + ["all"], width=2.7, sharey=True)
    fig.suptitle(name)
    for i, row_label in enumerate(names[row_key]):
        for j, col_label in enumerate(names[col_key]):
            aggregate_args = names.copy()
            aggregate_args[row_key] = row_label
            aggregate_args[col_key] = col_label
            y_true = aggregate_responses(y_true_all, *aggregate_args)
            plot_confidence_histogram(ax[i, j], y_true, n_bins, normalized=True)
    for i, row_label in enumerate(names[row_key]):
        aggregate_args = names.copy()
        aggregate_args[row_key] = row_label
        y_true = aggregate_responses(y_true_all, *aggregate_args)
        plot_confidence_histogram(ax[i, -1], y_true, n_bins=n_bins, normalized=True, color="tab:orange")
    for j, col_label in enumerate(names[col_key]):
        aggregate_args = names.copy()
        aggregate_args[col_key] = col_label
        y_true = aggregate_responses(y_true_all, *aggregate_args)
        plot_confidence_histogram(ax[-1, j], y_true, n_bins=n_bins, normalized=True, color="tab:orange")
    if save:
        save_fig(fig, f"{save}/{name}/accuracy_distribution.png")
    else:
        plt.show(fig)

def create_fig_confidence_distribution(name, y_pred_all, names, row_key, col_key, n_bins, save=None):
    fig, ax = create_subplots(names[row_key] + ["all"], names[col_key] + ["all"], width=2.7, sharey=True)
    fig.suptitle(name)
    for i, row_label in enumerate(names[row_key]):
        for j, col_label in enumerate(names[col_key]):
            aggregate_args = names.copy()
            aggregate_args[row_key] = row_label
            aggregate_args[col_key] = col_label
            y_pred = aggregate_responses(y_pred_all, *aggregate_args)
            plot_confidence_histogram(ax[i, j], y_pred, n_bins)
    for i, row_label in enumerate(names[row_key]):
        aggregate_args = names.copy()
        aggregate_args[row_key] = row_label
        y_pred = aggregate_responses(y_pred_all, *aggregate_args)
        plot_confidence_histogram(ax[i, -1], y_pred, n_bins=n_bins, normalized=True, color="tab:orange")
    for j, col_label in enumerate(names[col_key]):
        aggregate_args = names.copy()
        aggregate_args[col_key] = col_label
        y_pred = aggregate_responses(y_pred_all, *aggregate_args)
        plot_confidence_histogram(ax[-1, j], y_pred, n_bins=n_bins, normalized=True, color="tab:orange")
    if save:
        save_fig(fig, f"{save}/{name}/confidence_distribution.png")
    else:
        plt.show(fig)

def create_fig_calibration_curve(name, y_true_all, y_pred_all, names, row_key, col_key, n_bins, save=None):
    fig, ax = create_subplots(names[row_key] + ["all"], names[col_key] + ["all"], width=2.7)
    fig.suptitle(name)
    for i, row_label in enumerate(names[row_key]):
        for j, col_label in enumerate(names[col_key]):
            aggregate_args = names.copy()
            aggregate_args[row_key] = row_label
            aggregate_args[col_key] = col_label
            y_true = aggregate_responses(y_true_all, *aggregate_args)
            y_pred = aggregate_responses(y_pred_all, *aggregate_args)
            plot_calibration_curve(ax[i, j], y_true, y_pred, n_bins)
    for i, row_label in enumerate(names[row_key]):
        aggregate_args = names.copy()
        aggregate_args[row_key] = row_label
        y_true = aggregate_responses(y_true_all, *aggregate_args)
        y_pred = aggregate_responses(y_pred_all, *aggregate_args)
        plot_calibration_curve(ax[i, -1], y_true, y_pred, n_bins, color="tab:orange")
    for j, col_label in enumerate(names[col_key]):
        aggregate_args = names.copy()
        aggregate_args[col_key] = col_label
        y_true = aggregate_responses(y_true_all, *aggregate_args)
        y_pred = aggregate_responses(y_pred_all, *aggregate_args)
        plot_calibration_curve(ax[-1, j], y_true, y_pred, n_bins, color="tab:orange")
    fig.legend(["perfect calibration", "actual calibration"], bbox_to_anchor=(0.5, -0.07), loc="lower center", ncol=2)
    if save:
        save_fig(fig, f"{save}/{name}/calibration_curve.png")
    else:
        plt.show(fig)

def create_fig_calibration_ece(name, scores, row_labels, col_labels, save=None):
    fig, ax = plt.subplots(**get_figlayout(ncols=3, width=7, ratio=(len(row_labels), len(col_labels))), layout="constrained")
    fig.suptitle(name)
    ax[0].set_title("accuracy")
    plot_heatmap(ax[0], scores["accuracy"], row_labels, col_labels, vmin=0.5, vmax=1, cmap="Greens")
    ax[1].set_title("confidence")
    plot_heatmap(ax[1], scores["confidence"], row_labels, col_labels, vmin=0.5, vmax=1, cmap="Greens")
    ax[2].set_title("ECE")
    plot_heatmap(ax[2], scores["ece"], row_labels, col_labels, cmap="Reds")
    if save:
        save_fig(fig, f"{save}/{name}/calibration_ece.png")
    else:
        plt.show(fig)

def create_fig_informativeness_diversity(name, scores, row_labels, col_labels, save=None):
    fig, ax = plt.subplots(**get_figlayout(ncols=2, width=7, ratio=(len(row_labels), len(col_labels))), layout="constrained")
    fig.suptitle(name)
    ax[0].set_title("# distinct confidence scores")
    plot_heatmap(ax[0], scores["confidence_n_distinct"], row_labels, col_labels, cmap="Greens", format="{:.1f}")
    ax[1].set_title("variance of confidence scores")
    plot_heatmap(ax[1], scores["confidence_variance"], row_labels, col_labels, cmap="Greens")
    if save:
        save_fig(fig, f"{save}/{name}/informativeness_diversity.png")
    else:
        plt.show(fig)

def create_fig_meaningfulness_kldiv(name, scores, row_labels, col_labels, save=None):
    fig, ax = plt.subplots(**get_figlayout(width=7, ratio=(len(row_labels), len(col_labels))), layout="constrained")
    fig.suptitle(name)
    ax.set_title("KL-divergence( distr. || avg. distr. over datasets )")
    plot_heatmap(ax, scores["kl_div_over_dataset"], row_labels, col_labels, cmap="Oranges", vmin=0, vmax=0.5)
    if save:
        save_fig(fig, f"{save}/{name}/meaningfulness_kldiv.png")
    else:
        plt.show(fig)
