"""
The common input of the comparison: targets in the unwrapped (UW) domain, shared by both algorithms.

    registry.py   named targets (TARGETS), and the target file both algorithms read: outputs/runs/<run>/input/target.npy
    prism.py      rectangular prism: one 2D slice, (N_rho, N_l, 1) target
    tpms.py       TPMS lattice, bit-identical to the reference script's vamtoolbox devR2R buildtargetsR2R.tpms
"""
