# R2R tomographic VAM: ad hoc vs reference optimization

Roll-to-roll (R2R) computed axial lithography (article in `docs/`). Two algorithms compute the light
projections (sinogram) that print a target inside a photoresist web wrapped on a roller:

- **adhoc**: our implementation of the article's algorithm;
- **reference**: an existing script built from VAMToolbox (devR2R branch). Its algorithm is untouched; only its setup
  and input/output lines were edited (geometry, target file, output and matrix folders on the command line).

They run one after the other (reference, then adhoc) on **the same input target file**. Their results are compared
**symmetrically**: both write the same set of files, and the same metrics are applied to each.

```
                                     ┌──> algorithms/reference ──> <run>/reference/ ──┐
targets/ ──> <run>/input/target.npy ─┤                                                ├──> comparison/ ──> <run>/report.md
                                     └──> algorithms/adhoc     ──> <run>/adhoc/     ──┘
```

Every call of `main.py` is a new run with its own folder, `outputs/runs/<date>_<time>_<target>_z<slices>/`. Runs are
never overwritten. **To see the last run, open `outputs/runs/index.md`**: it lists every run, newest first, with the
loss and IoU of both algorithms, and links to each run's `report.md` (key results, checks, figures, all metrics).

## Run

```powershell
.venv\Scripts\python.exe main.py --target prism          # 2D prism slice (~3 min); or reference-tpms (default), gyroid-sheet, path\to\target.npy
.venv\Scripts\python.exe main.py --list-targets
.venv\Scripts\python.exe main.py --target prism --geometry tpms_diamond_skel6   # another reference geometry
.venv\Scripts\python.exe main.py --list-geometries
```

`--geometry` selects one of the reference script's geometry cases. It sets the material, roller and web configuration
(hence the propagation matrix, built automatically on first use) for both algorithms. With `--target reference-tpms`, it
also sets the script's own target, which `targets/tpms.py` reproduces only for the diamond geometries. A non-default
geometry is added to the run folder name: `<date>_<time>_<target>__<geometry>_z<slices>`.

Options: `--z-slices 16|1|all` for the 3D targets (default 16; the full target needs ~25 GB of RAM; `prism` is a
single 2D slice), `--n-iter 250`. Each step can also
run alone. The first one starts a run; the others work on the latest run, or on `--run <folder name>`:

```powershell
.venv\Scripts\python.exe -m targets.registry --target prism                # 1. start a run: the common input
.venv\Scripts\python.exe -m algorithms.reference.run
.venv\Scripts\python.exe -m algorithms.adhoc.run
.venv\Scripts\python.exe -m comparison.compare                            # report.md, then index.md
```

The reference script can also be run by hand, on any geometry and target:

```powershell
.venv\Scripts\python.exe algorithms\reference\original\xs_test_r2r_optimization_for_target_matrix.py tpms_bcc_skel0 --target-file t.npy --output-dir results
```

Without `--target-file` it builds its geometry's own target; without `--output-dir` it writes to
`outputs/reference_standalone/<config>/`.

## Layout

```
main.py                   entry point: target -> both algorithms -> comparison
paths.py                  all locations
targets/                  THE COMMON INPUT
    registry.py           named targets; starts a run: outputs/runs/<run>/input/target.npy
    prism.py, tpms.py     target generators
algorithms/
    adhoc/                ALGORITHM 1: ad hoc implementation of the article
        run.py            common target -> outputs/runs/<run>/adhoc
        propagator.py     shift-window R2R propagator (the article's algorithm) + BCLP glue
        BCLP.py           optimizer
    reference/            ALGORITHM 2: reference built from VAMToolbox
        run.py            common target -> outputs/runs/<run>/reference
        original/         the reference script (algorithm untouched; only setup and input/output lines edited)
                          and the vamtoolbox devR2R branch it imports
        config.py         reads the script's configuration (per geometry) without running it
        build_matrix.py   builds a missing propagation matrix in ~1 min (one z slice)
comparison/
    compare.py            metrics and figures from the two results folders -> report.md, index.md
docs/                     article
outputs/
    runs/
        index.md          ALL RUNS, newest first: target, geometry, z slices, loss and IoU of both algorithms
        <date>_<time>_<target>_z<slices>/
            report.md     THE RESULT: key numbers side by side, checks, figures, all metrics
            summary.json  the same, machine readable
            figures/      comparison figures shown in report.md
            input/        target.npy + target.json: the common input (target, geometry, z slices)
            reference/    reference results (.npy) and the script's own figures
            adhoc/        adhoc results (same .npy files), loss_history.npz, run_info.json
            logs/         reference.log, adhoc.log: console output of each algorithm
    matrices/             propagation matrices, one per geometry configuration, used by both algorithms
    reference_standalone/ results of the reference script run by hand (created when used)
```

Both algorithms use the same configuration (roller radius, web thickness, angular step, absorption, refractive
indices), taken from the reference script, and the same propagation matrix. Only the algorithms differ.
