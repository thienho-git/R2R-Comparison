"""
All project locations. Every comparison run gets its own folder, named after its start time so that the newest is
last, and outputs/runs/index.md lists them all (newest first) with their key results:

    outputs/runs/
        index.md                            one line per run: date, target, geometry, slices, loss and IoU of both
        <YYYY-MM-DD_HHMM>_<target>[__<geometry>]_z<slices>/
            report.md                       THE RESULT: checks, metrics of both algorithms side by side, figures
            summary.json                    the same, machine readable
            figures/                        comparison figures shown in report.md
            input/target.npy, target.json   the common input given to both algorithms (+ target, geometry, z slices)
            reference/                      reference results: target, sinogram, reconstruction, response .npy + its own figures
            adhoc/                          adhoc results: the same .npy files + loss_history.npz, run_info.json
            logs/                           reference.log, adhoc.log (console output of each algorithm)
    outputs/matrices/                       propagation matrices matrix_sino2uw_<config>.pt, shared by both algorithms
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

ALGORITHMS_DIR = ROOT/"algorithms"

# Reference algorithm: the original script (only its setup and input/output lines edited) and the devR2R vamtoolbox
REFERENCE_ORIGINAL_DIR = ALGORITHMS_DIR/"reference"/"original"
REFERENCE_SCRIPT = REFERENCE_ORIGINAL_DIR/"xs_test_r2r_optimization_for_target_matrix.py"
VAMTOOLBOX_DEVR2R = REFERENCE_ORIGINAL_DIR/"vamtoolbox_devR2R" #contains the vamtoolbox package of the devR2R branch

OUTPUT_DIR = ROOT/"outputs"
RUNS_DIR = OUTPUT_DIR/"runs"
MATRIX_DIR = OUTPUT_DIR/"matrices" #also the reference script's default --matrix-dir


def runDir(run):
    """Folder of a comparison run (see the module docstring)."""
    return RUNS_DIR/run


def runs():
    """Names of the run folders, oldest first (the names start with the date)."""
    return sorted(p.parent.parent.name for p in RUNS_DIR.glob("*/input/target.json")) if RUNS_DIR.exists() else []


def resolveRun(run=None):
    """A run folder name; None means the latest run."""
    if run is None:
        if not runs():
            raise FileNotFoundError(f"No run in {RUNS_DIR}. Start one with: python main.py --target T")
        return runs()[-1]
    run = Path(run).name #also accepts a path to the folder
    if not (runDir(run)/"input"/"target.json").exists():
        raise FileNotFoundError(f"No run '{run}' in {RUNS_DIR}. Runs: {runs()}")
    return run


def targetFile(run):
    """The common input of a run, given to both algorithms."""
    return runDir(run)/"input"/"target.npy"


def resultsDir(run, algorithm):
    """Results of one algorithm ('adhoc' or 'reference') in a run."""
    return runDir(run)/algorithm


def logFile(run, algorithm):
    return runDir(run)/"logs"/f"{algorithm}.log"


def referenceMatrix(alpha_mat):
    """Propagation matrix the reference script uses (built by algorithms.reference.build_matrix on the first run)."""
    matches = sorted(MATRIX_DIR.glob(f"matrix_sino2uw_alpha_mat={alpha_mat}_*.pt"))
    if not matches:
        raise FileNotFoundError(f"No matrix for alpha_mat={alpha_mat} in {MATRIX_DIR}.")
    return matches[0]


class TeeOutput:
    """Copy everything printed (stdout and stderr) to log_file as well, until exit."""
    _stream = _log = None

    def __init__(self, log_file):
        self.log_file = Path(log_file)

    def write(self, text):
        self._stream.write(text)
        if self._log is not None: #closed: a handler kept from inside the block (e.g. logging) still reaches the console
            self._log.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        if self._log is not None:
            self._log.flush()

    def __enter__(self):
        import sys
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_file, "a", encoding="utf-8", buffering=1)
        self._stream, self._stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self
        return self

    def __exit__(self, *exc):
        import sys
        if exc[0] is not None and not issubclass(exc[0], (SystemExit, KeyboardInterrupt)):
            import traceback
            self._log.write("".join(traceback.format_exception(*exc)))
        sys.stdout, sys.stderr = self._stream, self._stderr
        self._log.close()
        self._log = None
        return False

    def __getattr__(self, name): #isatty, encoding, reconfigure...
        return getattr(self._stream, name)


class KeepAwake:
    """Keep Windows from sleeping while a long run is in progress (process-scoped, released on exit)."""
    _ES_CONTINUOUS, _ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

    def __enter__(self):
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(self._ES_CONTINUOUS | self._ES_SYSTEM_REQUIRED)
        except (ImportError, AttributeError, OSError):
            pass
        return self

    def __exit__(self, *exc):
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(self._ES_CONTINUOUS)
        except (ImportError, AttributeError, OSError):
            pass
        return False
