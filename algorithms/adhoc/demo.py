"""
Standalone demo of the ad hoc algorithm (not part of the comparison): one z slice of a prism band in the unwrapped
domain -> R2R propagation with the ray-traced matrix in data/ -> BCLP -> unwrapped and wrapped plots.
Results: outputs/adhoc_demo.

    python -m algorithms.adhoc.demo
"""

import logging

from paths import ADHOC_DEMO_DIR, ADHOC_DEMO_MATRIX
from targets.prism import generatePrismSlice

from .create_geometry import createGeometry
from .plot_results import plotResults
from .run_optimization import runOptimization


def main():
    logging.basicConfig(level=logging.INFO)
    ADHOC_DEMO_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Target image: z slice of a rectangular prism, inside a band of finite width (the web) in the unwrapped domain
    image_file = generatePrismSlice(filename=str(ADHOC_DEMO_DIR/"prism_slice.png"), width=1, height=0.5, center=(0.0, -0.25))

    # 2) Geometry: UW target (band of length band_length cm, padded along the web) and R2R propagator built from the
    #    precomputed ray-traced matrix in data/ (absorption + refraction). The window (r_i, tau, deg_inc, N_rho, N_l) is
    #    read from the matrix file name and structure.
    target_geo, r2r_geo, propagator = createGeometry(target_file=image_file, band_length=2.0, matrix_file=str(ADHOC_DEMO_MATRIX))

    # 3) Optimization
    opt_sino, opt_recon, mapped_recon, logs = runOptimization(target_geo, propagator, n_iter=500, learning_rate=1e-4)

    # 4) Plot (unwrapped and wrapped)
    plotResults(target_geo, r2r_geo, opt_sino, mapped_recon, logs)


if __name__ == "__main__":
    main()
