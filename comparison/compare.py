"""
Compare the results of the two algorithms (algorithms/reference and algorithms/adhoc) on the same common target.

Checks and metrics:
    - the two targets are bit-identical;
    - the two propagators are the same operator: adhoc's back propagation of the reference sinogram is compared with
      the reference dose (both use the same matrix file);
    - for each method: BCLP loss with the reference settings (p = 3, eps 0.01/0.05, weight 3/1), share of voxels
      outside the tolerance band, response errors, dose separation between part and background, and the part printed
      by thresholding the response at 0.5 (IoU, Dice, false voxels);
    - between methods: relative differences and correlations of sinogram, dose and response.
Reads outputs/runs/<run>/{input,reference,adhoc}; writes outputs/runs/<run>/report.md (key results first, then all
metrics and the figures), summary.json and figures/, then updates the list of all runs, outputs/runs/index.md.

    python -m comparison.compare [--run RUN]     # default: the latest run
    python -m comparison.compare --index         # only rebuild outputs/runs/index.md
"""

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.adhoc.create_geometry import createGeometry
from algorithms.reference.config import referenceConfig
from paths import RUNS_DIR, logFile, referenceMatrix, resultsDir, runDir, runs
from targets.registry import addRunArgument, openRun

ALGORITHMS = ("reference", "adhoc")

P_NORM, EPS_IN, EPS_OUT, WEIGHT_IN = 3, 0.01, 0.05, 3 #reference script settings


def loadResults(folder):
    return {name: np.load(folder/f"{name}.npy", mmap_mode="r") for name in ("target", "sinogram", "reconstruction", "response")}


def volumeMetrics(target, dose, response):
    part = target > 0
    err = np.asarray(response, dtype=np.float64) - target
    eps = np.where(part, EPS_IN, EPS_OUT)
    band = np.abs(err) - eps
    v = band > 0
    weight = np.where(part, WEIGHT_IN, 1)
    loss = np.sum(v*weight*np.abs(band)**P_NORM)**(1/P_NORM)

    printed = response > 0.5
    tp = np.sum(printed & part)
    fp = np.sum(printed & ~part)
    fn = np.sum(~printed & part)
    dose_in, dose_out = dose[part], dose[~part]
    return {
        "bclp_loss": float(loss),
        "voxels_outside_band_fraction": float(v.mean()),
        "voxels_outside_band_fraction_part": float(v[part].mean()),
        "voxels_outside_band_fraction_background": float(v[~part].mean()),
        "response_error_mean_abs": float(np.abs(err).mean()),
        "response_error_rms": float(np.sqrt(np.mean(err**2))),
        "response_error_max_abs": float(np.abs(err).max()),
        "dose_part_min": float(dose_in.min()),
        "dose_part_p01": float(np.percentile(dose_in, 1)),
        "dose_part_mean": float(dose_in.mean()),
        "dose_background_mean": float(dose_out.mean()),
        "dose_background_p99": float(np.percentile(dose_out, 99)),
        "dose_background_max": float(dose_out.max()),
        "dose_contrast_p01_over_p99": float(np.percentile(dose_in, 1)/np.percentile(dose_out, 99)),
        "print_iou": float(tp/(tp + fp + fn)),
        "print_dice": float(2*tp/(2*tp + fp + fn)),
        "print_false_positive_voxels": int(fp),
        "print_false_negative_voxels": int(fn),
        "print_voxel_accuracy": float(1 - (fp + fn)/target.size),
    }


def sinogramMetrics(g, rows_per_chunk=64):
    total, total_sq, maximum, minimum, nonzero = 0.0, 0.0, -np.inf, np.inf, 0
    for r0 in range(0, g.shape[0], rows_per_chunk):
        block = np.asarray(g[r0:r0 + rows_per_chunk], dtype=np.float64)
        total += block.sum()
        total_sq += np.sum(block**2)
        maximum, minimum = max(maximum, block.max()), min(minimum, block.min())
        nonzero += np.count_nonzero(block)
    middle_angle = np.asarray(g[:, g.shape[1]//2, :])
    return {"shape": list(g.shape), "dtype": str(g.dtype), "sum": float(total), "l2": float(np.sqrt(total_sq)),
            "max": float(maximum), "min": float(minimum), "nonzero_fraction": float(nonzero/g.size),
            "distinct_levels_middle_angle": int(np.unique(middle_angle).size)}


def pairMetrics(a, b, rows_per_chunk=64):
    """Relative L2 difference ||a - b||/||b|| and Pearson correlation, chunked over the first axis."""
    n = s_a = s_b = s_aa = s_bb = s_ab = s_dd = 0.0
    for r0 in range(0, a.shape[0], rows_per_chunk):
        x = np.asarray(a[r0:r0 + rows_per_chunk], dtype=np.float64).ravel()
        y = np.asarray(b[r0:r0 + rows_per_chunk], dtype=np.float64).ravel()
        n += x.size
        s_a, s_b = s_a + x.sum(), s_b + y.sum()
        s_aa, s_bb, s_ab = s_aa + x @ x, s_bb + y @ y, s_ab + x @ y
        s_dd += np.sum((x - y)**2)
    cov = s_ab/n - (s_a/n)*(s_b/n)
    corr = cov/np.sqrt((s_aa/n - (s_a/n)**2)*(s_bb/n - (s_b/n)**2))
    return {"relative_l2_difference": float(np.sqrt(s_dd/s_bb)), "correlation": float(corr)}


def propagatorCheck(reference, config):
    """Adhoc back propagation of the reference sinogram vs the reference dose (same matrix, float64)."""
    _, _, propagator = createGeometry(target=np.zeros((config.N_rho, config.N_l, 1)), matrix_file=str(referenceMatrix(config.alpha_mat)),
                                      pad_target=False, propagator_kwargs=dict(dtype=torch.float64, z_batch=16))
    dose = propagator.backward(np.asarray(reference["sinogram"]))
    return pairMetrics(dose, reference["reconstruction"]), propagator.adjointTest()


def plotComparison(target, results, config, out_dir):
    z = target.shape[2]//2
    extent = [0, target.shape[1]/config.f_l*10, config.r_o*10, config.r_i*10] #mm, row 0 = r_i on top like the reference plots
    names = list(results)

    fig, axs = plt.subplots(3, 2, figsize=(16, 10))
    axs[0, 0].imshow(target[:, :, z], cmap="gray", aspect="auto", extent=extent, vmin=0, vmax=1)
    axs[0, 0].set_title(f"Target (same for both), z slice {z}")
    for ax, name in zip(axs[1:, 0], names):
        im = ax.imshow(results[name]["response"][:, :, z], cmap="gray", aspect="auto", extent=extent, vmin=0, vmax=1)
        ax.set_title(f"{name}: response M(f)")
        fig.colorbar(im, ax=ax)
    diff = np.asarray(results[names[1]]["response"][:, :, z], float) - results[names[0]]["response"][:, :, z]
    im = axs[0, 1].imshow(diff, cmap="RdBu_r", aspect="auto", extent=extent, vmin=-1, vmax=1)
    axs[0, 1].set_title(f"Response difference ({names[1]} - {names[0]})")
    fig.colorbar(im, ax=axs[0, 1])
    for ax, name in zip(axs[1:, 1], names):
        err = np.asarray(results[name]["response"][:, :, z], float) - target[:, :, z]
        im = ax.imshow(err, cmap="RdBu_r", aspect="auto", extent=extent, vmin=-1, vmax=1)
        ax.set_title(f"{name}: response error M(f) - target")
        fig.colorbar(im, ax=ax)
    for ax in axs.ravel():
        ax.set(xlabel="l along web (mm)", ylabel="rho (mm)")
    fig.tight_layout()
    fig.savefig(out_dir/"responses_middle_slice.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    part = target > 0
    for ax, name in zip(axs, names):
        dose = np.asarray(results[name]["reconstruction"])
        bins = np.linspace(0, np.percentile(dose, 99.9), 200)
        ax.hist(dose[~part], bins=bins, alpha=0.6, density=True, label="background")
        ax.hist(dose[part], bins=bins, alpha=0.6, density=True, label="part")
        ax.set(title=f"{name}: dose distribution", xlabel="dose", ylabel="density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir/"dose_histograms.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sinos = [np.asarray(results[name]["sinogram"][:, :, z], float) for name in names]
    vmax = max(s.max() for s in sinos)
    for ax, name, sino in zip(axs, names, sinos):
        im = ax.imshow(sino, aspect="auto", cmap="CMRmap", vmin=0, vmax=vmax)
        ax.set(title=f"{name}: sinogram, z slice {z}", xlabel="angle / time step", ylabel="detector")
        fig.colorbar(im, ax=ax)
    im = axs[2].imshow(sinos[1] - sinos[0], aspect="auto", cmap="RdBu_r")
    axs[2].set(title=f"Sinogram difference ({names[1]} - {names[0]})", xlabel="angle / time step", ylabel="detector")
    fig.colorbar(im, ax=axs[2])
    fig.tight_layout()
    fig.savefig(out_dir/"sinograms_middle_slice.png", dpi=150)
    plt.close(fig)


def referenceElapsed(run):
    """Optimization time of the reference script: the last 'Elapsed time: N seconds' it printed to its log."""
    log = logFile(run, "reference")
    times = re.findall(r"Elapsed time: ([\d.]+) seconds", log.read_text(encoding="utf-8", errors="replace")) if log.exists() else []
    return float(times[-1]) if times else None


def _seconds(value):
    return "?" if value is None else f"{value:.0f} s"


def writeReport(summary, path):
    volume, names = summary["volume"], list(summary["volume"])
    info = summary["target_info"]
    z = info["z_slices"]
    slices = f"{len(z)} z slice{'s'*(len(z) > 1)}" + ("" if len(z) == info["full_shape"][2] else f" {z}")

    def row(label, cell):
        return f"| {label} | " + " | ".join(cell(n) for n in names) + " |"

    lines = [f"# {summary['run']}", "",
             f"Target **{info['name']}**, geometry `{info['geometry']}`, {slices} of {tuple(info['full_shape'])}, "
             f"started {info.get('created', '?')}.", "",
             "| | " + " | ".join(names) + " |", "|---|" + "---|"*len(names),
             row("BCLP loss (lower is better)", lambda n: f"**{volume[n]['bclp_loss']:.4g}**"),
             row("IoU of the printed part (response > 0.5)", lambda n: f"**{volume[n]['print_iou']:.4g}**"),
             row("voxels printed outside / missing in the part", lambda n: f"{volume[n]['print_false_positive_voxels']} / {volume[n]['print_false_negative_voxels']}"),
             row("mean dose in the part / background", lambda n: f"{volume[n]['dose_part_mean']:.3g} / {volume[n]['dose_background_mean']:.3g}"),
             row("optimization time", lambda n: _seconds(summary["elapsed_s"].get(n))), "",
             "Checks:",
             f"- same input target for both (bit-identical): **{summary['target_identical']}**",
             f"- same propagator: adhoc back propagation of the reference sinogram vs reference dose, relative L2 difference "
             f"{summary['propagator_check']['relative_l2_difference']:.2e}, correlation {summary['propagator_check']['correlation']:.6f} "
             f"(adhoc adjoint test {summary['adhoc_adjoint_test']:.1e})", "",
             "## Figures (middle z slice)", "",
             "![responses](figures/responses_middle_slice.png)", "",
             "![dose histograms](figures/dose_histograms.png)", "",
             "![sinograms](figures/sinograms_middle_slice.png)", "",
             "Loss histories: ![adhoc loss](figures/adhoc_loss.png) ![reference loss](reference/optimizationplot.png)", "",
             "## All metrics", "",
             "| metric | " + " | ".join(names) + " |", "|---|" + "---|"*len(names)]
    for key in volume[names[0]]:
        lines.append(row(key, lambda n: f"{volume[n][key]:.4g}"))
    lines += ["", "| sinogram | " + " | ".join(names) + " |", "|---|" + "---|"*len(names)]
    for key in summary["sinogram"][names[0]]:
        lines.append(row(key, lambda n: str(summary["sinogram"][n][key])))
    lines += ["", "Between methods (relative L2 difference w.r.t. the reference, correlation):", ""]
    for key, value in summary["between_methods"].items():
        lines.append(f"- {key}: {value['relative_l2_difference']:.3g}, correlation {value['correlation']:.4f}")
    lines += ["", f"Configuration: `{summary['config']}`", "",
              "Files: `input/` (common target), `reference/` and `adhoc/` (results of each algorithm, same .npy files), "
              "`logs/` (console output of each algorithm), `summary.json` (this report)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def writeIndex():
    """outputs/runs/index.md: every run, newest first, with its status and key results."""
    lines = ["# Runs (newest first)", "",
             "Each folder holds `report.md` (the result), `input/`, `reference/`, `adhoc/`, `logs/`. "
             "Loss: BCLP loss, lower is better. IoU: printed part vs target, 1 is perfect.", "",
             "| run | target | geometry | z slices | reference loss / IoU | adhoc loss / IoU |", "|---|---|---|---|---|---|"]
    for run in reversed(runs()):
        folder = runDir(run)
        info = json.loads((folder/"input"/"target.json").read_text())
        n_z = len(info["z_slices"])
        cells = [f"[{run}]({run}/report.md)" if (folder/"report.md").exists() else run, info["name"].split("__")[0],
                 info["geometry"], f"{n_z} of {info['full_shape'][2]}"]
        summary = json.loads((folder/"summary.json").read_text()) if (folder/"summary.json").exists() else None
        for algorithm in ALGORITHMS:
            if summary is not None:
                v = summary["volume"][algorithm]
                cells.append(f"{v['bclp_loss']:.4g} / {v['print_iou']:.3g}")
            else:
                cells.append("done, not compared" if (resultsDir(run, algorithm)/"response.npy").exists() else "not run / failed")
        lines.append("| " + " | ".join(cells) + " |")
    (RUNS_DIR/"index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Index of runs: {RUNS_DIR/'index.md'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    addRunArgument(parser)
    parser.add_argument("--index", action="store_true", help="only rebuild outputs/runs/index.md")
    args = parser.parse_args(argv)
    if args.index:
        writeIndex()
        return
    run, common_target, target_info = openRun(args.run)
    folder = runDir(run)
    figures_dir = folder/"figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    adhoc_dir = resultsDir(run, "adhoc")

    config = referenceConfig()
    results = {algorithm: loadResults(resultsDir(run, algorithm)) for algorithm in ALGORITHMS}
    target = np.asarray(common_target, dtype=np.float64)

    summary = {"run": run, "target": target_info["name"], "config": str(config), "target_info": target_info,
               "results_dirs": {algorithm: str(resultsDir(run, algorithm)) for algorithm in ALGORITHMS},
               "target_shape": list(target.shape),
               "target_identical": all(bool(np.array_equal(results[algorithm]["target"], target)) for algorithm in ALGORITHMS)}
    print(f"Both algorithms ran on the common target: {summary['target_identical']}")

    summary["propagator_check"], summary["adhoc_adjoint_test"] = propagatorCheck(results["reference"], config)
    print(f"Propagator check: {summary['propagator_check']}")

    summary["volume"] = {name: volumeMetrics(target, np.asarray(r["reconstruction"]), np.asarray(r["response"])) for name, r in results.items()}
    summary["sinogram"] = {name: sinogramMetrics(r["sinogram"]) for name, r in results.items()}
    summary["between_methods"] = {key: pairMetrics(results["adhoc"][key], results["reference"][key])
                                  for key in ("sinogram", "reconstruction", "response")}

    info_file = adhoc_dir/"run_info.json"
    summary["adhoc_run"] = json.loads(info_file.read_text()) if info_file.exists() else None
    summary["elapsed_s"] = {"reference": referenceElapsed(run), "adhoc": (summary["adhoc_run"] or {}).get("elapsed_s")}
    loss_file = adhoc_dir/"loss_history.npz"
    if loss_file.exists():
        loss = np.load(loss_file)["loss"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(loss)
        ax.set(title="Ad hoc BCLP loss", xlabel="iteration", ylabel="loss", yscale="log")
        fig.tight_layout()
        fig.savefig(figures_dir/"adhoc_loss.png", dpi=150)
        plt.close(fig)

    plotComparison(target, results, config, figures_dir)
    (folder/"summary.json").write_text(json.dumps(summary, indent=2))
    writeReport(summary, folder/"report.md")
    print((folder/"report.md").read_text(encoding="utf-8").split("## Figures")[0])
    print(f"Report: {folder/'report.md'}")
    writeIndex()


if __name__ == "__main__":
    main()
