"""Plot manually recorded rate-distortion data; no training logs are read automatically."""

import matplotlib.pyplot as plt
import numpy as np
import os

experiments = {
    "gaussian_only": {
        "Ours": {
            "rate":  np.array([1.0, 0.8331, 0.6133, 0.5289, 0.4122, 0.2996, 0.2403]),
            "psnr":  np.array([34.0127, 34.0528, 33.9988, 33.8980, 33.257, 27.23, 24.46]),
            "ssim":  np.array([0.962, 0.962, 0.961, 0.960, 0.956, 0.927, 0.903]),
            "lpips": np.array([0.0831, 0.0833, 0.0847, 0.086, 0.0913, 0.1245, 0.1543]),
        },

        "RDOGS-based": {
            "rate":  np.array([1.0, 0.9, 0.6221, 0.5037, 0.3927, 0.2929, 0.2195]),
            "psnr":  np.array([34.0127, 33.9177, 32.52, 31.0685, 29.6637, 27.9207, 25.498]),
            "ssim":  np.array([0.962, 0.962, 0.950, 0.935, 0.917, 0.894, 0.864]),
            "lpips": np.array([0.0831, 0.0831, 0.106, 0.139, 0.172, 0.2074, 0.25]),
        },
    },

    "sh_only": {
        "Ours": {
            "rate":  np.array([1.0, 0.69, 0.40, 0.27, 0.2043, 0.1677, 0.1397, 0.04]),
            "psnr":  np.array([34.0127, 33.9059, 33.9197, 33.9466, 33.9361, 33.8320, 33.5172, 30.9371]),
            "ssim":  np.array([0.962, 0.9619, 0.962, 0.9621, 0.9620, 0.9617, 0.9612, 0.9567]),
            "lpips": np.array([0.0831, 0.0836, 0.0835, 0.0835, 0.08367, 0.08413, 0.08460, 0.08867]),
        },

    }
}

markers = ['o', 's', '^', 'D', 'v', '*', 'x', 'P']
linestyles = ['-', '--', '-.', ':']

metric_info = {
    "psnr":  {"ylabel": "PSNR (dB)", "higher_better": True},
    "ssim":  {"ylabel": "SSIM",      "higher_better": True},
    "lpips": {"ylabel": "LPIPS",     "higher_better": False},
}

def check_data_validity(experiments):
    """Require equal array lengths within each method before plotting."""
    for exp_name, methods in experiments.items():
        for method_name, data in methods.items():
            lengths = {k: len(v) for k, v in data.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"[{exp_name} - {method_name}] 各数组长度不一致: {lengths}"
                )

def plot_rd_for_experiment(exp_name, methods, metric_name, save_dir="figures", use_log_x=True):
    """Plot one metric from the manually entered experiment table and save a PNG.

    Rates <= 0.01 are omitted. The SH-only plot always uses a linear rate axis;
    other experiments use a log axis when use_log_x is True.
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plotted_metrics = []

    for i, (method, data) in enumerate(methods.items()):
        rate = data["rate"]
        metric = data[metric_name]

        idx = np.argsort(rate)
        rate = rate[idx]
        metric = metric[idx]

        mask = rate > 0.01
        rate = rate[mask]
        metric = metric[mask]

        plotted_metrics.extend(metric.tolist())

        plt.plot(
            rate,
            metric,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2,
            markersize=6,
            label=method
        )

    if use_log_x and exp_name != "sh_only":
        plt.xscale('log')

    plt.xlabel('Rate')
    plt.ylabel(metric_info[metric_name]["ylabel"])
    plt.title(f'{exp_name} - RD Curve ({metric_info[metric_name]["ylabel"]})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if exp_name == "sh_only" and len(plotted_metrics) > 0:
        y_min = min(plotted_metrics)
        y_max = max(plotted_metrics)
        y_center = 0.5 * (y_min + y_max)

        min_span_map = {
            "psnr": 1.0,
            "ssim": 0.002,
            "lpips": 0.003,
        }

        span = max(y_max - y_min, min_span_map[metric_name])
        pad = 0.08 * span
        plt.ylim(y_center - span / 2 - pad, y_center + span / 2 + pad)

    plt.tight_layout()

    filename = os.path.join(save_dir, f'{exp_name}_{metric_name}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


if __name__ == "__main__":
    check_data_validity(experiments)

    for exp_name, methods in experiments.items():
        for metric_name in ["psnr", "ssim", "lpips"]:
            plot_rd_for_experiment(exp_name, methods, metric_name, save_dir="figures")


    print("All figures are saved in ./figures/")
