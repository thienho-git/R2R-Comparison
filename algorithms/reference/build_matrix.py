"""
Build the single-slice propagation matrix P_UW->g that the reference script loads from outputs/matrices/matrix_sino2uw_*.pt.

Why: when that file is missing, the reference script ray-traces it on its full 3D target (N_r x N_z = 306800 rays for
the TPMS target), which runs for hours and then exceeds 8 GB of GPU memory in coalesce(). The matrices actually used
(e.g. matrix_sino2uw_alpha_mat=2.2_...pt, 1475 rows) are single-slice: the reference propagator reuses one slice
matrix for every z. This script makes that matrix with the same code path and settings as the script
(TargetGeometryR2R, homogeneous_r2r attenuation/absorption/index models, eikonal ray tracing at 269.9 deg,
PyTorchRayTracingPropagator.buildPropagationMatrix), on one z slice of the target.

The configuration (alpha_mat, n_sur, n_mat, r_i, tau, deg_inc, ...) is read from the reference script itself: its
configr2r class and the `case` of the selected geometry are extracted with ast and executed, nothing else of the script
is run. The devR2R vamtoolbox is put first on sys.path before vamtoolbox is imported. run.py calls this
automatically when the matrix is missing; standalone:

    python -m algorithms.reference.build_matrix [--geometry NAME] [--alpha-mat 2.2] [--out file.pt]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from paths import MATRIX_DIR, VAMTOOLBOX_DEVR2R

from .config import addGeometryArgument, applyGeometry, referenceConfig


def buildMatrix(config, out_file):
    if str(VAMTOOLBOX_DEVR2R) not in sys.path:
        sys.path.insert(0, str(VAMTOOLBOX_DEVR2R)) #devR2R vamtoolbox before the main-branch one of the venv
    import torch
    import vamtoolbox as vam
    if not vam.__file__.startswith(str(VAMTOOLBOX_DEVR2R)):
        raise ImportError(f"vamtoolbox must be the devR2R branch ({VAMTOOLBOX_DEVR2R}), found {vam.__file__}.")

    #One z slice: the matrix only depends on the grid and the optical models, not on the target values
    target = np.zeros((config.N_rho, config.N_l, 1))
    target_geo = vam.geometry.TargetGeometryR2R(target=target, r_i=config.r_i, tau=config.tau,
                                                 spatial_sampling_rate_rho=config.f_rho, spatial_sampling_rate_l=config.f_l)
    coord_vec = target_geo.constructCoordVec()

    #Same models and ray-tracing options as the reference script
    model_args = dict(rho_grid=target_geo.transform.rho_g_w, r_mid=target_geo.transform.r_mid, tau=target_geo.transform.tau)
    attenuation_model = vam.medium.AttenuationModel(coord_vec, type="interpolation", form="homogeneous_r2r", alpha_internal=config.alpha_mat, **model_args)
    absorption_model = vam.medium.AttenuationModel(coord_vec, type="interpolation", form="homogeneous_r2r", alpha_internal=config.alpha_mat, **model_args)
    index_model = vam.medium.IndexModel(coord_vec, type="interpolation", form="homogeneous_r2r", n_sur=config.n_sur, n=config.n_mat, **model_args)
    proj_geo_rt = vam.geometry.ProjectionGeometry(np.array([269.9]), ray_type="ray_trace", CUDA=True,
                                                  index_model=index_model, attenuation_model=attenuation_model,
                                                  absorption_model=absorption_model, ray_trace_method="eikonal",
                                                  eikonal_parametrization="physical_path_length",
                                                  ray_trace_ode_solver="forward_symplectic_euler",
                                                  ray_trace_ray_config="r2r", inclination_angle=0.0, ray_density=1)
    P_rt = vam.projector.pyTorchRayTraceR2R.PyTorchRayTracingPropagator(target_geo, proj_geo_rt, output_torch_tensor=True)

    start = time.time()
    P = P_rt.buildPropagationMatrix()
    print(f"Built {tuple(P.shape)} matrix with {P.coalesce().values().numel()} nonzeros in {time.time() - start:.0f} s")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(P, out_file)
    return out_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha-mat", type=float, default=None, help="override alpha_mat (cm^-1), e.g. 2.2")
    parser.add_argument("--out", default=None, help="output .pt (default: the file name the reference script looks for)")
    addGeometryArgument(parser)
    args = parser.parse_args(argv)
    applyGeometry(args)

    config = referenceConfig()
    if args.alpha_mat is not None:
        config.alpha_mat = args.alpha_mat
    out_file = MATRIX_DIR/f"matrix_sino2uw_{config.matrixStr()}.pt" if args.out is None else args.out
    print(f"Reference configuration: {config}")
    print(f"Building the single-slice matrix -> {out_file}")
    buildMatrix(config, Path(out_file))


if __name__ == "__main__":
    main()
