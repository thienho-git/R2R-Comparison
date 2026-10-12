# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Capstone project on roll-to-roll (R2R) tomographic volumetric additive manufacturing (computed axial lithography, CAL). Goals:
1. Build a working ad hoc implementation of the article's algorithm in `algorithms/adhoc/`.
2. Compare it, symmetrically and on the same chosen input target, with a reference algorithm built from VAMToolbox elements: `algorithms/reference/original/xs_test_r2r_optimization_for_target_matrix.py`. **Never modify the algorithm of the reference script or the devR2R vamtoolbox** (`algorithms/reference/original/`). Its setup and input/output lines may be edited when needed (paths, target and geometry selection, compatibility): prefer such direct edits to workarounds around the script.

The article is in `docs/R2R_tomographic_volumetric_additive_manufacturing_article_UCB.pdf`. The relevant parts are Fig. 2, the "Numerical light propagation model" section, and supplementary sections S2, S3 and S11. No PDF text tool is installed in `.venv`. To read the article, extract it with `pypdf` installed to a scratch location, not into the project venv.

Layout (`README.md` has the tree):
- `main.py`: entry point. Common target → reference → adhoc → compare.
- `paths.py`: every location (inputs, originals, outputs).
- `targets/`: the common input. `registry.py` (named targets, `prepareTarget`/`loadTarget`), `prism.py`, `tpms.py`.
- `algorithms/adhoc/`: ad hoc algorithm. `run.py` (comparison run), `demo.py` (standalone single-slice demo with plots), `data/` (ray-traced α = 2.2 matrix used by the demo), algorithm modules.
- `algorithms/reference/`: reference algorithm. `original/` (the script + `vamtoolbox_devR2R/`, the VAMToolbox `devR2R` branch at commit d63e924), `run.py`, `config.py`, `build_matrix.py`.
- `comparison/compare.py`: metrics and figures from the two results folders → the run's `report.md`; rebuilds `outputs/runs/index.md`.
- `outputs/` (generated): `runs/index.md` (every run, newest first, with loss/IoU of both), `runs/<YYYY-MM-DD_HHMM>_<target>[__<geometry>]_z<slices>/` (one folder per run, never overwritten: `report.md`, `summary.json`, `figures/`, `input/target.{npy,json}`, `reference/`, `adhoc/`, `logs/{reference,adhoc}.log`), `matrices/` (propagation matrices `matrix_sino2uw_<config>.pt`, used by both algorithms), `adhoc_demo/`, `reference_standalone/` (the script run by hand without `--output-dir`).
- Runs: `targets.registry.prepareTarget` starts a run (new folder, `input/target.npy`, geometry saved in `input/target.json`). The later steps (`algorithms.reference.run`, `algorithms.adhoc.run`, `comparison.compare`) share one interface: `--run RUN` (default: the latest run, `paths.resolveRun`); `registry.openRun` loads the target and selects the run's geometry. Both `run.py` write `target`/`sinogram`/`reconstruction`/`response` `.npy` files to `<run>/<algorithm>/` and tee their console to `<run>/logs/<algorithm>.log` (`paths.TeeOutput`).

Windows only: `algorithms/adhoc/BCLP.py` imports `msvcrt`. Pressing `q` during optimization stops the gradient-descent loop.

## Commands

The environment is a Python 3.12 venv at `.venv`. Dependencies are in `requirements.txt`. torch is the CUDA 12.8 build (cu128 index), which the RTX 50xx GPU requires. The main-branch vamtoolbox (used by adhoc) is installed from its GitHub zip because it is not on PyPI. The devR2R branch is not installed (same package name): the reference script puts `original/vamtoolbox_devR2R` first on `sys.path` itself.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py [--target reference-tpms|prism|gyroid-sheet|file.npy] [--z-slices 16|1|all] [--n-iter 250]   # a new run; result: outputs/runs/index.md -> <run>/report.md
.venv\Scripts\python.exe main.py --list-targets
.venv\Scripts\python.exe main.py --target prism --geometry tpms_diamond_skel6   # --list-geometries lists the script's cases
.venv\Scripts\python.exe -m targets.registry --target prism   # each step alone: start a run, then algorithms.reference.run, algorithms.adhoc.run, comparison.compare (latest run, or --run RUN)
.venv\Scripts\python.exe -m algorithms.adhoc.demo   # band target (UW) -> R2R propagation -> BCLP -> UW and wrapped plots
.venv\Scripts\python.exe algorithms\reference\original\xs_test_r2r_optimization_for_target_matrix.py [GEOMETRY] [--target-file t.npy] [--output-dir d] [--matrix-dir m]   # the reference script by hand
```

All modules import from the project root (`paths`, `targets`, `algorithms`, `comparison`). Run them with `python -m` from the root, not as standalone scripts.

There is no test suite, linter or build step. `R2RAlgebraicPropagator.adjointTest()` checks that `backward` is the exact adjoint of `forward`. Run it after any change to the propagator.

## Architecture

Ad hoc demo pipeline (`algorithms/adhoc/demo.py`), with every step working in the article's **unwrapped (UW)** domain: a flat web with axes ρ (thickness, rows, ρ = 0 is the inner face r_i, where the light enters) and l (along the web, columns).
1. `targets.prism.generatePrismSlice` writes a band image (`outputs/adhoc_demo/prism_slice.png`). The band is the whole image; it is bounded by the web thickness and not cropped to a cylinder.
2. `create_geometry` builds `R2RGeometry` from physical parameters (r_i, tau, deg_inc, n_rho, straight_length) and the propagator. It resamples the image to N_ρ × round(band_length / d_l) voxels, then zero-pads it by N_l − 1 on both ends along l, so N_θ = N_long − N_l + 1. It also takes a voxel array (`target=`, used as is, e.g. `targets.tpms.generateTPMSTarget`) and `pad_target=False`, where the propagator pads implicitly like the reference: only the target voxels are optimized, with the same N_θ = N_obj + N_l − 1.
   - **The demo uses the precomputed ray-traced matrix in `algorithms/adhoc/data/`** (passed as `matrix_file=`). It includes absorption and refraction. The window (r_i, tau, deg_inc, N_ρ, N_l) is read from the file name and the matrix structure, so `createGeometry`'s physical-parameter arguments are ignored.
   - Without `matrix_file=`, `createGeometry` falls back to the vamtoolbox standard projector (straight rays only). `standard_projector.buildUWMatrixFromVamtoolbox` takes the explicit single-angle sparse matrix of `vamtoolbox.projectorconstructor` (CPU → `Projector3DParallelSparse.A`, clipped to the inscribed circle) on a square W grid that contains the whole window, and composes it with T into `P_UW→g = P_W→g @ T` (about 2 s to build). It is verified against `projector.forward()` of the wrapped window to about 1e-15. vamtoolbox integrates along array **axis 1** at angle 0; the builder detects this and reorders T to match.
3. `run_optimization` builds the vamtoolbox `Options` and calls `minimizeR2RBCLP`.
4. `plot_results` draws the UW target and dose, the sinogram, and the same target and dose wrapped (W) onto the roller at one time step.

The band is wrapped onto the roller only through the propagation matrix. Light propagation happens in the **wrapped (W)** domain, with the web curved around the roller.

`algorithms/adhoc/r2r_wrapping.py` is the core:
- The optional precomputed composite single-angle matrix `P_UW→g` (`algorithms/adhoc/data/matrix_sino2uw_*.pt`, and the reference's matrices in `outputs/matrices/`) is a torch sparse COO tensor of shape 1475 × 149552 = N_r × (N_ρ·N_l), with N_ρ = 208 and N_l = 719. The column index is `i_rho*N_l + i_l`. The filename encodes the physical parameters (lengths in cm, `deg_inc` in degrees). `loadPropagationMatrix` reads it with `torch.load(weights_only=True)`.
- `R2RAlgebraicPropagator` (torch, sparse CSR on CUDA when available, else CPU; processes `z_batch` z-slices at a time; numpy in/out of `dtype`, float64 by default, so `BCLP.py` stays numpy; `implicit_padding` as above): forward unfolds the long UW tomogram into shifting windows (one window per angle/time step; the web advances one l-voxel per `deg_inc`), multiplies by P, and produces a sinogram. Backward applies Pᵀ and folds the windows back by summing overlaps.
- `R2RGeometry`: the explicit T: UW ↔ W transform (180° wrap around the roller plus two straight runs). It is used for the wrapped plots, for converting targets, and for `composeUWMatrix` (P_UW = P_W @ T). Its conventions follow the reference (devR2R `transformationR2R.WrapTransformation`): ρ row 0 = r_i, l pitch d_l = r_i·Δθ (66 µm), straight runs r_i·π/2. Before, they were inferred from the matrix and used r_o and mid-thickness (79 µm).
- `minimizeR2RBCLP` passes the R2R propagator to `BCLP.BCLPNorm(..., projector=propagator)`, which becomes `self.P`. With `projector=None`, BCLP falls back to the vamtoolbox standard projector, as in conventional CAL. It also passes a minimal `UWTargetGeometry` (anything with `.array`).
- `plot_results` `time_index` is a roller position (sinogram column), not an iteration. The default is the middle column (509 of 0–1019 for the 2 cm band), which centers the band on the roller bottom. The dose shown is always the final iterate.

### Interface contract BCLP.py relies on
- The projector must provide `forward(f)` and `backward(g)`, and optionally `inverseBackward(f, filter)` for the initial sinogram. BCLP's gradient is `P.forward(...)`, so `backward` must be the exact adjoint of `forward`.
- Sinograms are kept as `(detector, angle, z)`, called "cylindrical" in the code. The optimizer flattens them to 1D for the update and reshapes them back using `g0_shape` (`checkSinogramShape`). Discretization with `bit_depth` works per angle along **axis 1**.
- `BCLP.py` reads option attributes that `vamtoolbox.optimize.Options` may not define (`momentum`, `regularize_l`, `g0`, `blb`, `bub`, `bit_depth`, `exit_param`, `save_img_path`). `run_optimization.py` fills in these defaults.

### Gotchas
- Learning rate depends on which matrix is used:
  - vamtoolbox projector: 1e-6 is monotone, 3e-6 is the default (faster, slightly noisy), and 1e-5 or higher oscillates.
  - `data/` matrix (used by the demo): 1e-4, monotone in the first iterations (about 0.09 s per iteration on the RTX 5050 GPU for N_θ ≈ 1000; it was 1.4 s with scipy on CPU).
- Cost with the vamtoolbox projector (measured before the torch port, scipy on CPU): about 2 s per iteration for the default 2 cm band (N_θ ≈ 970, P is 1469 × 149136 with about 0.74 M nonzeros).
- Comments tagged `[WRAP NOTE]` in `BCLP.py` use "wrap/unwrap" to mean only array reshaping. Comments tagged `[W/UW NOTE]` (written when the pipeline was conventional CAL) use the article's physical W/UW domains. The two are unrelated.
- vamtoolbox's own `pyTorchAlgebraicPropagation` is the conventional single-matrix P·f. It has no shift window and requires torch.

## Comparison with the reference

- **Reference script edits** (setup and I/O only, algorithm untouched): `original/vamtoolbox_devR2R` first on `sys.path`; `np.product`/`np.NaN` restored (removed in NumPy 2, used by devR2R); an `argparse` block (`parse_known_args`, so an IDE kernel's arguments are ignored) for `geometry` (positional, default `tpms_diamond_skel0`), `--target-file` (`.npy` used instead of the case's own target, also in the two `.mat` cases), `--output-dir` (results folder, default `outputs/reference_standalone/<config>`) and `--matrix-dir` (default `outputs/matrices`); the hard-coded `D:\` paths replaced by those folders; the two diagnostics at fixed angles 719 and 1000 use the middle angle.
- `algorithms/reference/run.py` calls the script with the geometry, `--target-file <run>/input/target.npy`, `--output-dir <run>/reference`, `--matrix-dir outputs/matrices`, and `MPLBACKEND=Agg` so `plt.show()` doesn't block. Console output goes to `<run>/logs/reference.log`.
- **Common target (`targets/registry.py`):** one voxel array on the sampling of the selected geometry's configuration (N_ρ × N_l × N_z = 208 × 753 × 208), saved as `<run>/input/target.npy` (after the z subset) with `target.json` (name, geometry, config, z slices). Both algorithms read that file. Named targets are in `TARGETS` (`reference-tpms`, the default and the script's own TPMS rebuilt bit for bit by `targets/tpms.py`; `prism`, a 2D image of one prism slice, shape (N_ρ, N_l, 1), so `--z-slices` does not apply; `gyroid-sheet`); a `.npy` path also works (target name `file-<stem>`). To add one, write a function `config -> array` and register it.
- **Geometry (`--geometry`):** selects a `case` of the script.
  - `config.py` holds the selection (`setGeometry`, default read from the script's `add_argument("geometry", default=...)`). The geometry is saved with the run's target; the later steps read it from there (`openRun`), and `run.py` passes it to the script.
  - The geometry sets α, n, r_i, τ, deg_inc, hence the matrix (`build_matrix.py` builds a missing one in about 1 min). Run folder names get `__<geometry>` after the target when it is not the default.
  - `reference-tpms` is reproducible only for diamond geometries: bcc-skel is not in `targets/tpms.py`, cubic-skel uses `p_cubic`/`shrink_z`, and two cases load `.mat` files that are not available. Every geometry works with the other targets.
- **Memory:** the full reference target is 208 × 753 × 208. Its BCLP needs about 25 GB of RAM (float64 sinograms of 1475 × 1471 × 208), so the default is `--z-slices 16` (evenly spaced, indices saved in `target.json`). Propagation is slice-independent; the global loss and quantization still couple the slices.
- **Matrix:** if the matrix is missing, the script ray-traces it on the full 3D target: about 306 k rays, hours, then GPU out of memory. `algorithms/reference/build_matrix.py` builds it instead with the same ray tracer and settings on one z-slice (about 1 min). Rebuilding α = 2.2 this way matches `algorithms/adhoc/data/` with a 0.5% relative Frobenius difference and the same sparsity.
- **Config and target:** `algorithms/reference/config.py` reads the configuration and TPMS parameters from the reference script with `ast` (only the config statements run). `targets/tpms.py` reproduces the target bit for bit.
- **`algorithms/adhoc/run.py`:** same common target, matrix, implicit padding and optimization settings as the script (B = 10, p = 3, eps 0.01/0.05, weight 3/1, lr 0.05, momentum 0.5, 8 bit, 250 iterations), float32 propagation.
- **Remaining algorithm differences:**
  - devR2R BCLP quantizes the whole sinogram to `bit_depth` every iteration. adhoc `BCLP.py` quantizes per angle every 10 iterations.
  - The propagators differ in implementation only: adhoc's back-propagation of the reference sinogram matches the reference dose to 3e-7.
- **Result of the 16-slice run on `reference-tpms` (`outputs/runs/2026-10-10_1711_reference-tpms_z16/`):**
  - The reference loss went 114 → 110 → about 125 (part and background equally overdosed, IoU 0.21). adhoc went 113 → 10.8 (IoU 1.0 at a response threshold of 0.5).
  - Swapping devR2R's quantization rule into adhoc's BCLP reproduces the reference failure (loss about 119, part dose = background dose). No quantization behaves like adhoc's rule.
  - So the gap comes from devR2R's global per-iteration 8-bit quantization, not from the target, matrix or propagator.
- **Result of the 1-slice run on `prism` (`outputs/runs/2026-10-10_2002_prism_z1/`):** same pattern. Reference loss 47.2, IoU 0.20, background dose (5.76) above part dose (2.12). adhoc loss 3.8, IoU 1.0. (An earlier 16-slice prism run, reference 94.3 / 0.20 vs adhoc 8.0 / 1.0, was overwritten under the old layout.)
- **`comparison/compare.py`** writes `<run>/report.md` (key numbers of both side by side, checks, embedded figures, all metrics), `summary.json` and `figures/`, then rebuilds `outputs/runs/index.md` (also `--index` alone). The reference time is the script's own `Elapsed time` line in its log.
