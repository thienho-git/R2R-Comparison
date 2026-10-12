"""
Composite R2R propagation matrix built from the vamtoolbox standard projector instead of the precomputed matrix in data/.

Article (Numerical light propagation model):  P_W->g f_W = g,   T_W->UW(P_W->g) f_UW = g,   P_UW->g f_UW = g.
Here P_W->g is vamtoolbox's standard parallel-beam projector (vamtoolbox.projectorconstructor) at a single angle on a
square Cartesian grid centered on the roller axis, and T: UW -> W is R2RGeometry.uwToWMatrix, so that
    P_UW->g = P_W->g @ T
The result plugs into R2RAlgebraicPropagator (shift window over the web) exactly like the precomputed matrix.

Differences with the precomputed data/ matrix: the standard projector traces straight rays without absorption or
refraction (vamtoolbox's absorption option models a rotating cylindrical vial, not a web on a roller), and its
detector count/pitch follow the W grid (one detector per grid column, trimmed to the detectors that see the web).
"""

import numpy as np
import scipy.sparse
import vamtoolbox


def wGridForProjector(r2r_geo, pixel=None):
    """
    Square W grid for the standard projector: centered on the roller axis, pixel = d_rho by default, and large enough
    that the inscribed circle (vamtoolbox clips to it) contains the whole window, straight runs included.
    Returns x = y (n,) pixel-center coordinates, increasing.
    """
    pixel = r2r_geo.d_rho if pixel is None else pixel
    corners_rho, corners_l = np.meshgrid([0, r2r_geo.n_rho - 1], [0, r2r_geo.n_l - 1, r2r_geo.l_center], indexing="ij")
    x, y = r2r_geo.uwToXY(corners_rho, corners_l)
    radius = np.max(np.hypot(x, y)) + 2*pixel
    half = int(np.ceil(radius/pixel))
    return (np.arange(2*half + 1) - half)*pixel


def _rayAxis(projector, n):
    #Which array axis the single-angle projection integrates along: moving a pixel along that axis leaves the detector unchanged.
    probe = np.zeros((n, n, 1))
    probe[n//2, n//2] = 1
    base = np.argmax(projector.forward(probe)[:, 0, 0])
    probe[:] = 0
    probe[n//2 + n//4, n//2] = 1
    return 0 if np.argmax(projector.forward(probe)[:, 0, 0]) == base else 1


def _singleAngleMatrix(projector, n):
    """
    Explicit P_W->g (n_det x n*n, columns in C order of the (n, n) target array) of a vamtoolbox projector at one angle.
    Uses the projector's own sparse matrix and circle clipping when available (Projector3DParallelSparse, the standard
    CPU projector), otherwise probes backward() with one unit detector at a time.
    """
    mask = vamtoolbox.util.data.clipToCircle(np.ones((n, n, 1)))[:, :, 0].ravel()
    if hasattr(projector, "A"):
        return scipy.sparse.csr_matrix(projector.A, dtype=np.float64) @ scipy.sparse.diags(mask)

    n_det = projector.forward(np.zeros((n, n, 1))).shape[0]
    rows = []
    for d in range(n_det):
        unit = np.zeros((n_det, 1, 1))
        unit[d] = 1
        rows.append(scipy.sparse.csr_matrix(np.asarray(projector.backward(unit)).reshape(1, -1)))
    return scipy.sparse.vstack(rows).tocsr()


def buildUWMatrixFromVamtoolbox(r2r_geo, angle=0.0, pixel=None, cache_file=None):
    """
    Build P_UW->g = P_W->g @ T with P_W->g from vamtoolbox.projectorconstructor (standard projector, CPU, parallel rays).

    Parameters
    ----------
    r2r_geo : R2RGeometry
        Window geometry (T: UW -> W). Its layout has the beam along y, so the projector ray axis is matched to the
        y axis of the grid automatically.
    angle : float
        Projection angle (deg) passed to vamtoolbox. Only its ray axis matters here; it is checked against the grid.
    pixel : float, optional
        W grid pixel size (default d_rho).
    cache_file : str or Path, optional
        .npz file to reuse the matrix between runs (rebuilt if missing).

    Returns
    -------
    P : scipy.sparse.csr_matrix (N_det x N_rho*N_l), detectors that see no part of the window removed.
    """
    if cache_file is not None:
        try:
            return scipy.sparse.load_npz(cache_file).tocsr()
        except (OSError, FileNotFoundError):
            pass

    grid = wGridForProjector(r2r_geo, pixel)
    n = grid.size
    target_geo = vamtoolbox.geometry.TargetGeometry(target=np.zeros((n, n)), clip_to_circle=False)
    proj_geo = vamtoolbox.geometry.ProjectionGeometry(np.array([angle], dtype=float), ray_type="parallel", CUDA=False)
    projector = vamtoolbox.projectorconstructor.projectorconstructor(target_geo, proj_geo)

    P_w = _singleAngleMatrix(projector, n)
    #T rows follow the (y, x) C order of the grid; with rays along array axis 0 the array rows are y (beam direction)
    T = r2r_geo.uwToWMatrix(grid, grid)
    if _rayAxis(projector, n) == 1:
        order = np.arange(n*n).reshape(n, n).T.ravel() #array (i, j) <-> grid (y=j, x=i)
        T = T[order, :]
    P = (P_w @ T).tocsr()
    P.eliminate_zeros()

    seen = np.flatnonzero(P.getnnz(axis=1))
    P = P[seen[0]:seen[-1] + 1, :].tocsr()

    if cache_file is not None:
        scipy.sparse.save_npz(cache_file, P)
    return P
