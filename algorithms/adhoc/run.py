"""
Run the ad hoc algorithm on the common target, with the same problem as the reference:
    - target: the common target of the run, outputs/runs/<run>/input/target.npy (targets/registry.py). It is checked
      to be identical to the reference target.npy when the reference has been run;
    - configuration and propagation matrix: those of the run's geometry, the same file as the reference
      (outputs/matrices/matrix_sino2uw_<config>.pt);
    - padding: implicit, like the reference propagator (only the target voxels are optimized; N_theta = N_l_target + N_l - 1);
    - optimization: the settings of the reference script (lines 428-493): response model B=10, g0 = forward propagation
      of map_inv(target) / (N_theta*100), weight 3 in the part and 1 outside, eps 0.01 in the part and 0.05 outside,
      p = 3, learning rate 0.05, momentum 0.5, 8-bit sinogram, 250 iterations; float32 propagation like the reference.
What stays different is the algorithm itself: adhoc's R2RAlgebraicPropagator and BCLP.py (see comparison/compare.py).
Results: outputs/runs/<run>/adhoc; console output: outputs/runs/<run>/logs/adhoc.log.

    python -m algorithms.adhoc.run [--run RUN] [--n-iter 250]     # default: the latest run
"""

import argparse
import json
import logging
import sys
import time

import numpy as np
import torch
import vamtoolbox

from algorithms.reference.config import referenceConfig
from paths import KeepAwake, TeeOutput, logFile, referenceMatrix, resultsDir
from targets.registry import addRunArgument, openRun

from .propagator import R2RAlgebraicPropagator, minimizeR2RBCLP


def bclpOptions(**kwargs):
    """
    vamtoolbox Options for BCLP, completed with the options read by the local BCLP.py that vamtoolbox may not define.
    kwargs are passed to vamtoolbox.optimize.Options (method, n_iter, p, q, eps, weight, learning_rate, g0, ...).
    """
    options = vamtoolbox.optimize.Options(**{"method": "BCLP", **kwargs})
    defaults = {"momentum": 0.0, "regularize_l": 0.0, "g0": None, "blb": 0, "bub": None, "bit_depth": None,
                "exit_param": None, "save_img_path": None}
    for key, value in defaults.items():
        if not hasattr(options, key):
            setattr(options, key, value)
    return options


def checkTargetAgainstReference(name, target):
    """When the reference has already run, its saved target must be the common target."""
    try:
        reference = np.load(resultsDir(name, "reference")/"target.npy", mmap_mode="r")
    except FileNotFoundError:
        print("Reference target.npy not found yet: the target identity is checked by compare.py.")
        return None
    identical = reference.shape == target.shape and bool(np.array_equal(reference, target))
    print(f"Target identical to the reference target.npy: {identical} (shape {target.shape})")
    if not identical:
        raise ValueError("The common target differs from the reference target.npy (different target or z slices): "
                         "rerun the reference with the same --target and --z-slices.")
    return identical


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    addRunArgument(parser)
    parser.add_argument("--n-iter", type=int, default=250, help="iterations (reference: 250)")
    args = parser.parse_args(argv)
    name, target, _ = openRun(args.run)
    with TeeOutput(logFile(name, "adhoc")):
        run(name, target, args.n_iter)


def run(name, target, n_iter):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True) #BCLP.py logs one line per iteration
    out_dir = resultsDir(name, "adhoc")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = referenceConfig()
    print(f"Reference configuration: {config}, common target '{name}' {target.shape}")
    identical = checkTargetAgainstReference(name, target)

    matrix_file = referenceMatrix(config.alpha_mat)
    print(f"Propagation matrix: {matrix_file.name}")
    propagator = R2RAlgebraicPropagator.fromMatrixFile(matrix_file, dtype=torch.float32)
    n_theta = propagator.nAngles(target.shape[1])

    #Optimization settings of the reference script
    response_model = vamtoolbox.response.ResponseModel(B=10)
    g0 = propagator.forward(response_model.map_inv(target))
    g0 /= n_theta*100
    weight = np.ones_like(target)
    eps_map = np.ones_like(target)
    eps_map[target > 0] = 0.01
    eps_map[target == 0] = 0.05
    weight[target > 0] = 3
    options = bclpOptions(n_iter=n_iter, filter="ram-lak", verbose="time", p=3, learning_rate=0.05, regularize_l=0,
                          momentum=0.5, response_model=response_model, eps=eps_map, g0=g0, weight=weight, bit_depth=8)
    print(f"g0 {g0.shape} {g0.dtype}, N_theta = {n_theta}, device {propagator.device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with KeepAwake():
        opt_sino, dose, mapped_dose, logs = minimizeR2RBCLP(target, propagator, options)
    elapsed = time.time() - start
    peak_gpu_mb = torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None
    print(f"Elapsed time: {elapsed:.1f} s, peak GPU memory {peak_gpu_mb:.0f} MB")

    np.save(out_dir/"target.npy", target)
    np.save(out_dir/"sinogram.npy", opt_sino)
    np.save(out_dir/"reconstruction.npy", dose)
    np.save(out_dir/"response.npy", mapped_dose)
    np.savez(out_dir/"loss_history.npz", loss=logs.loss, iter_times=logs.iter_times, l0=logs.l0, l1=logs.l1, l2=logs.l2,
             lf=logs.lf, l0_eps0=logs.l0_eps0, l1_eps0=logs.l1_eps0, l2_eps0=logs.l2_eps0, lf_eps0=logs.lf_eps0)
    run_info = dict(run=name, config=str(config), matrix=matrix_file.name, target_identical_to_reference=identical,
                    n_iter=n_iter, n_theta=n_theta, elapsed_s=elapsed, peak_gpu_mb=peak_gpu_mb, propagator_dtype=str(propagator.dtype))
    (out_dir/"run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Adhoc results: {out_dir}")


if __name__ == "__main__":
    main()
