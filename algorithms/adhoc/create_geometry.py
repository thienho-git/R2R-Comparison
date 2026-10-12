import logging
import os

import numpy as np

from .r2r_wrapping import R2RAlgebraicPropagator, R2RGeometry, UWTargetGeometry, imageToUWTarget, loadPropagationMatrix, padTarget
from .standard_projector import buildUWMatrixFromVamtoolbox


def createGeometry(target_file=None, band_length=2.0, r_i=0.76, tau=0.3, deg_inc=0.5, n_rho=208, straight_length=None,
                   matrix_file=None, cache_file=None, target=None, pad_target=True, propagator_kwargs=None, **geometry_kwargs):
    """
    Build the unwrapped (UW) target and the R2R propagation system.

    The target is either an image (target_file) or a voxel array (target):
    - image: a band of finite width (the web). Its rows are resampled to the N_rho voxels across the web thickness tau
      and its columns to round(band_length/d_l) voxels along the web, as a single z slice.
    - array: used voxel for voxel, without resampling (e.g. targets.tpms.generateTPMSTarget, the reference
      target). Its first axis must already be the N_rho of the window.
    Nothing is cropped to a cylinder; the band is wrapped onto the roller only through the propagation (and
    R2RGeometry for display).

    Propagation: by default P_UW->g is built from the vamtoolbox standard projector (standard_projector.py), with the
    window defined by the physical parameters below. If matrix_file is given, the precomputed matrix is used instead
    and the window is read from it (r_i, tau, deg_inc, n_rho, straight_length ignored).

    Parameters
    ----------
    target_file : str
        Band image (.png), rows = rho (thickness), columns = l (along the web), e.g. from targets.prism.generatePrismSlice.
    band_length : float
        Length of the target band along the web (cm), image targets only.
    r_i, tau : float
        Roller radius and web thickness (cm).
    deg_inc : float
        Angular increment per time step (deg); the web advances one l voxel per increment.
    n_rho : int
        Voxels across the web thickness (sets d_rho and the W grid pixel).
    straight_length : float, optional
        Straight web length simulated on each side of the wrap (cm). Default: reference quarter turn r_i*pi/2.
    matrix_file : str, optional
        Precomputed P_UW->g (.pt or .npz) to use instead of the vamtoolbox projector.
    cache_file : str, optional
        .npz cache for the vamtoolbox-built matrix (must correspond to the same parameters).
    target : ndarray (N_rho, N_obj[, N_z]), optional
        Voxel target used as is, in place of target_file.
    pad_target : bool
        True (default): the target is zero-padded by N_l - 1 voxels on both ends along l and the padding is part of
        the optimization (dose there is driven to 0). False: the target array is kept as is and the propagator pads
        implicitly, like the reference algorithm, so only the target voxels are optimized.
    propagator_kwargs : dict, optional
        Passed to R2RAlgebraicPropagator (e.g. dtype=torch.float32, device, chunk, z_batch).
    geometry_kwargs
        Passed to R2RGeometry (e.g. rho0_face, r_ref).

    Returns
    -------
    target_geo : UWTargetGeometry
        Target of shape (N_rho, N_long, N_z). Number of angles N_theta = propagator.nAngles(N_long)
        (= N_obj + N_l - 1 in both padding modes).
    r2r_geo : R2RGeometry
    propagator : R2RAlgebraicPropagator
    """
    if (target_file is None) == (target is None):
        raise ValueError("Give exactly one of target_file and target.")
    if matrix_file is not None:
        P = loadPropagationMatrix(matrix_file)
        r2r_geo = R2RGeometry.fromMatrixFile(matrix_file, P=P, **geometry_kwargs)
        source = f"data matrix ({os.path.basename(matrix_file)})"
    else:
        r2r_geo = R2RGeometry.fromStraightLength(r_i, tau, deg_inc, n_rho, straight_length, **geometry_kwargs)
        P = buildUWMatrixFromVamtoolbox(r2r_geo, cache_file=cache_file)
        r2r_geo.n_r = P.shape[0]
        source = "vamtoolbox standard projector"
    logging.getLogger(__name__).info(f"Projector: {source} | P_UW->g {P.shape[0]} x {P.shape[1]}, "
                                     f"window N_rho x N_l = {r2r_geo.n_rho} x {r2r_geo.n_l}")
    propagator = R2RAlgebraicPropagator(P, r2r_geo.n_rho, r2r_geo.n_l, implicit_padding=not pad_target, **(propagator_kwargs or {}))

    if target is None:
        n_band = max(1, int(round(band_length/r2r_geo.d_l)))
        band = imageToUWTarget(target_file, r2r_geo.n_rho, n_band)
    else:
        band = np.atleast_3d(np.asarray(target, dtype=np.float64))
        if band.shape[0] != r2r_geo.n_rho:
            raise ValueError(f"Target has {band.shape[0]} rho voxels but the window has N_rho = {r2r_geo.n_rho}.")
    band = np.atleast_3d(band)
    target_geo = UWTargetGeometry(padTarget(band, r2r_geo.n_l) if pad_target else band)

    return target_geo, r2r_geo, propagator
