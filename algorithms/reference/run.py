"""
Run the reference algorithm on the common target of a run: original/xs_test_r2r_optimization_for_target_matrix.py,
called as

    python xs_test_r2r_optimization_for_target_matrix.py GEOMETRY --target-file outputs/runs/<run>/input/target.npy
        --output-dir outputs/runs/<run>/reference --matrix-dir outputs/matrices

with MPLBACKEND=Agg so that its plt.show() calls do not block. When the propagation matrix of the geometry is missing,
build_matrix.py builds it first (same ray tracer and settings as the script, on one z slice: about 1 min, where the
script would ray-trace the full 3D target for hours and run out of GPU memory).

GEOMETRY is the one saved with the run's target. The console output goes to outputs/runs/<run>/logs/reference.log.

    python -m algorithms.reference.run [--run RUN]     # default: the latest run (start one with python -m targets.registry)
"""

import argparse
import os
import subprocess
import sys
import time

from paths import MATRIX_DIR, REFERENCE_SCRIPT, ROOT, KeepAwake, TeeOutput, logFile, resultsDir, targetFile
from targets.registry import addRunArgument, openRun

from .config import activeGeometry, referenceConfig


def run(args, cwd):
    """Run `python args...`, echoing its output."""
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    print(f"Running {' '.join(args)}", flush=True)
    start = time.time()
    process = subprocess.Popen([sys.executable] + args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in process.stdout:
        if "psf sinogram pixel" in line: #one line per detector while the matrix is built
            continue
        sys.stdout.write(line)
    process.wait()
    print(f"Exit code {process.returncode} after {time.time() - start:.0f} s", flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed (exit code {process.returncode}).")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    addRunArgument(parser)
    args = parser.parse_args(argv)
    run_name, _, _ = openRun(args.run)
    results_dir = resultsDir(run_name, "reference")
    results_dir.mkdir(parents=True, exist_ok=True)

    with TeeOutput(logFile(run_name, "reference")), KeepAwake():
        print(f"Reference algorithm on run {run_name}, geometry {activeGeometry()}")
        matrix_file = MATRIX_DIR/f"matrix_sino2uw_{referenceConfig().matrixStr()}.pt"
        if not matrix_file.exists():
            run(["-m", "algorithms.reference.build_matrix", "--geometry", activeGeometry(), "--out", str(matrix_file)], cwd=ROOT)
        run([str(REFERENCE_SCRIPT), activeGeometry(), "--target-file", str(targetFile(run_name)),
             "--output-dir", str(results_dir), "--matrix-dir", str(MATRIX_DIR)], cwd=results_dir)
        print(f"Reference results: {results_dir}")


if __name__ == "__main__":
    main()
