"""
The two R2R CAL algorithms compared on the same common target (targets/). Each one has a run.py with the same
interface, `python -m algorithms.<name>.run [--run RUN]` (default: the latest run), which reads
outputs/runs/<run>/input/target.npy and writes the same files (target, sinogram, reconstruction, response .npy) to
outputs/runs/<run>/<name>/:

    adhoc/       ad hoc implementation of the article's algorithm (docs/)
    reference/   reference algorithm built from VAMToolbox devR2R elements (original/: algorithm untouched, only setup and input/output lines edited)
"""
