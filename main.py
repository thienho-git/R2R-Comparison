"""
R2R tomographic VAM: compare two optimization algorithms on the same input target.

    common target (targets/) --+--> reference algorithm (algorithms/reference) --+--> comparison (comparison/)
                               +--> ad hoc algorithm    (algorithms/adhoc)     --+

    python main.py [--target reference-tpms|prism|gyroid-sheet|file.npy] [--geometry NAME] [--z-slices 16|all] [--n-iter 250]
    python main.py --list-targets
    python main.py --list-geometries

--geometry selects one of the reference script's geometry cases (material, roller and web configuration, and the
script's own target for --target reference-tpms). Default: the script's default, "tpms_diamond_skel0".

Each call is a new run, in its own folder outputs/runs/<YYYY-MM-DD_HHMM>_<target>[__<geometry>]_z<slices>/; its
result is report.md in that folder. outputs/runs/index.md lists every run, newest first (see paths.py).
Each step can also be run on its own; the steps after the first work on the latest run (or --run RUN):
    python -m targets.registry --target T [--z-slices N] [--geometry G]    # start a run: the common target
    python -m algorithms.reference.run
    python -m algorithms.adhoc.run
    python -m comparison.compare
"""

import argparse
import sys

from algorithms.reference import config
from targets import registry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default=registry.DEFAULT_TARGET,
                        help=f"common target: one of {sorted(registry.TARGETS)} or a .npy path (default {registry.DEFAULT_TARGET})")
    parser.add_argument("--list-targets", action="store_true", help="list the named targets and exit")
    config.addGeometryArgument(parser)
    parser.add_argument("--list-geometries", action="store_true", help="list the reference script's geometries and exit")
    parser.add_argument("--z-slices", default="16", help="evenly spaced z slices of the target, or 'all' (needs ~25 GB of RAM)")
    parser.add_argument("--n-iter", default="250", help="ad hoc iterations (the reference script uses 250)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if args.list_targets:
        registry.main(["--list"])
        return
    if args.list_geometries:
        default = config.defaultGeometry()
        print("\n".join(f"{name}{' (default)' if name == default else ''}" for name in config.geometries()))
        return
    config.applyGeometry(args)

    #Imported here so that --list-targets does not load torch and vamtoolbox
    from algorithms.adhoc import run as run_adhoc
    from algorithms.reference import run as run_reference
    from comparison import compare

    run = ["--run", registry.prepareTarget(args.target, args.z_slices)[0]]
    try:
        run_reference.main(run)
        run_adhoc.main(run + ["--n-iter", args.n_iter])
    except BaseException:
        compare.writeIndex() #also lists the failed run
        raise
    compare.main(run) #report.md, then the index


if __name__ == "__main__":
    main()
