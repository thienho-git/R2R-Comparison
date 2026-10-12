"""
Reference algorithm: an R2R CAL optimization script built from VAMToolbox devR2R elements.

    original/        the script and the devR2R vamtoolbox it needs. In the script, only setup and input/output lines
                     were edited (devR2R on sys.path, NumPy 2 aliases, command-line geometry/target/output/matrix
                     paths, middle angle in two diagnostics); never edit its algorithm.
    run.py           runs the script on the common target -> outputs/runs/<run>/reference
    config.py        reads the script's configuration without running it
    build_matrix.py  builds the script's propagation matrix on one z slice (instead of hours on the full target)
"""
