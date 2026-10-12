"""
The common input of the comparison: one target voxel array (N_rho, N_l, N_z), given unchanged to both algorithms.

It is built on the sampling of the reference script's selected geometry (configr2r, read by
algorithms/reference/config.py: r_i, tau, deg_inc, f_rho, f_l -> N_rho x N_l x N_z), so that the propagation matrix
both algorithms use applies to it. Choose it by name (TARGETS below) or give a path to a .npy array whose first axis
is N_rho:

    python main.py --target prism
    python main.py --target path/to/target.npy
    python main.py --list-targets

Preparing a target starts a new run: outputs/runs/<date>_<target>[__<geometry>]_z<slices>/input/target.npy (see paths.py).
Both algorithms read that same file: adhoc loads it, and the reference script gets it as --target-file. The selected
geometry is saved with it (input/target.json), so the later steps of the run use it without being told again. The configuration (alpha_mat, n, r_i, tau, deg_inc), and therefore the propagation matrix, is the
one of the script's selected geometry. "reference-tpms" is the script's own TPMS target, rebuilt bit for bit.

To add a target, write a function config -> array (float64, 0/1, shape (config.N_rho, N_l, config.N_z)) and add it
to TARGETS.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms.reference.config import activeGeometry, addGeometryArgument, applyGeometry, defaultGeometry, referenceSpec, setGeometry
from paths import RUNS_DIR, resolveRun, runDir, targetFile

from .prism import generatePrismImage
from .tpms import R2RSampling, generateTPMSTarget

DEFAULT_TARGET = "reference-tpms"


def _sampling(config):
    sampling = R2RSampling(r_i=config.r_i, tau=config.tau, l_max=config.l_max, deg_inc=config.deg_inc)
    if sampling.shape != (config.N_rho, config.N_l, config.N_z) or sampling.f_l != config.f_l or sampling.f_rho != config.f_rho:
        raise ValueError(f"targets.tpms sampling {sampling.shape} differs from the reference ({config.N_rho}, {config.N_l}, {config.N_z}).")
    return sampling


def referenceTPMS(config):
    """The reference script's own target (TPMS of its selected geometry), rebuilt bit for bit."""
    tpms = referenceSpec()[1]
    if tpms is None:
        raise NotImplementedError(f"Geometry '{activeGeometry()}' of the reference script does not build a TPMS target.")
    if tpms["unsupported"]:
        raise NotImplementedError(f"tpms arguments {tpms['unsupported']} of geometry '{activeGeometry()}' are not reproduced by targets/tpms.py.")
    return generateTPMSTarget(_sampling(config), unit_cell_size=tpms["unit_cell_size"], t=tpms["t"], kind=tpms["kind"])


def prism(config):
    """2D image (one z slice) of a rectangular prism spanning the whole web along l, half its thickness."""
    return generatePrismImage((config.N_rho, config.N_l), width=1, height=0.5)[:, :, None]


def gyroidSheet(config):
    """Gyroid sheet lattice, unit cell = web thickness tau."""
    return generateTPMSTarget(_sampling(config), unit_cell_size=config.tau, t=(0.4,), kind="gyroid-sheet")


# name -> builder(config)
TARGETS = {
    "reference-tpms": referenceTPMS,
    "prism": prism,
    "gyroid-sheet": gyroidSheet,
}


def targetName(target):
    """
    Name of a target argument: a TARGETS name, or 'file-<stem>' for a .npy path, followed by '__<geometry>' when the
    reference geometry is not the script's default.
    """
    if target in TARGETS:
        name = target
    elif target.endswith(".npy"):
        name = f"file-{Path(target).stem}"
    else:
        raise ValueError(f"Unknown target '{target}'. Choose one of {sorted(TARGETS)} or give a .npy path.")
    geometry = activeGeometry()
    return name if geometry == defaultGeometry() else f"{name}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', geometry)}"


def newRunName(name, n_slices):
    """<YYYY-MM-DD_HHMM>_<name>_z<slices>, unique in outputs/runs."""
    run = f"{datetime.now():%Y-%m-%d_%H%M}_{name}_z{n_slices}"
    suffix = 2
    while (RUNS_DIR/run).exists():
        run = f"{datetime.now():%Y-%m-%d_%H%M}_{name}_z{n_slices}_{suffix}"
        suffix += 1
    return run


def zSlices(n_z, z_slices):
    """Evenly spaced z indices (one slice: the middle one), or None for all slices."""
    if z_slices in (None, "all") or int(z_slices) >= n_z:
        return None
    if int(z_slices) == 1:
        return [n_z//2]
    return sorted(set(np.linspace(0, n_z - 1, int(z_slices)).round().astype(int).tolist()))


def buildTarget(target, config):
    if target in TARGETS:
        array = TARGETS[target](config)
    else:
        array = np.load(target)
    array = np.atleast_3d(np.asarray(array, dtype=np.float64))
    if array.ndim != 3 or array.shape[0] != config.N_rho:
        raise ValueError(f"Target shape {array.shape}: expected (N_rho = {config.N_rho}, N_l, N_z).")
    return array


def prepareTarget(target, z_slices="16"):
    """
    Start a new run: build the common target, keep z_slices evenly spaced z slices ('all' for the full target), and
    save it with its description in the run's input/ folder. Returns (run, array, info).
    """
    config = referenceSpec()[0]
    name = targetName(target)
    array = buildTarget(target, config)
    z_index = zSlices(array.shape[2], z_slices)
    run = newRunName(name, array.shape[2] if z_index is None else len(z_index))
    info = dict(run=run, name=name, source=str(Path(target).resolve()) if name.startswith("file-") else name,
                created=f"{datetime.now():%Y-%m-%d %H:%M}", full_shape=list(array.shape),
                z_slices=list(range(array.shape[2])) if z_index is None else z_index, geometry=activeGeometry(),
                config=str(config))
    if z_index is not None:
        array = np.ascontiguousarray(array[:, :, z_index])
    info["shape"] = list(array.shape)

    target_file = targetFile(run)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_file, array)
    target_file.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"New run: {runDir(run)}")
    print(f"Common target '{name}': {array.shape} (full {tuple(info['full_shape'])})")
    return run, array, info


def openRun(run=None):
    """
    The common target of a run (None: the latest one): (run, array, info). Also selects the run's geometry, so that
    the reference configuration and matrix are the ones the target was built for.
    """
    run = resolveRun(run)
    info = json.loads(targetFile(run).with_suffix(".json").read_text())
    setGeometry(info["geometry"])
    return run, np.load(targetFile(run)), info


def addRunArgument(parser):
    """--run option of the steps that work on an existing run."""
    parser.add_argument("--run", default=None, help="run folder name in outputs/runs (default: the latest run)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list the named targets")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="target name or .npy path to prepare")
    parser.add_argument("--z-slices", default="16", help="evenly spaced z slices, or 'all'")
    addGeometryArgument(parser)
    args = parser.parse_args(argv)
    applyGeometry(args)
    if args.list:
        for name, builder in TARGETS.items():
            print(f"{name:16s} {builder.__doc__.strip()}{' (default)' if name == DEFAULT_TARGET else ''}")
        return
    prepareTarget(args.target, args.z_slices)


if __name__ == "__main__":
    main()
