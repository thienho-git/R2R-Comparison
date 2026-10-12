"""
TPMS lattice target in the unwrapped (UW) domain, identical to the one of the reference algorithm
(algorithms/reference/original/xs_test_r2r_optimization_for_target_matrix.py: configr2r + vamtoolbox devR2R util.buildtargetsR2R.tpms).

The sampling and the implicit surfaces are written with the same floating-point operations in the same order as the
reference, so the voxel array is bit-identical (checked by comparison/compare.py against the reference target.npy, target "reference-tpms").
The array is built one z slice at a time, which gives the same values with ~1/N_z of the memory.

Array layout (N_rho, N_l, N_z): rows rho through the web (row 0 = r_i, as in the reference transformationR2R),
columns l along the web, z across the web width.
"""

import numpy as np


class R2RSampling:
    """
    Voxel sampling of the reference script (configr2r). Lengths in cm, deg_inc in degrees.
        f_rho = 1/(10.8 um DMD pixel * 1.33 magnification)   voxels/cm through the thickness and along z
        f_l   = 1/(deg_inc * r_i)                             voxels/cm along the web (one voxel per angular increment
                                                              at the inner radius)
    """

    def __init__(self, r_i=0.76, tau=0.3, l_max=5.0, deg_inc=0.5, dmd_pixel=0.00108, magnification=1.33):
        self.r_i = r_i
        self.tau = tau
        self.r_o = r_i + tau
        self.l_max = l_max
        self.deg_inc = deg_inc
        self.f_rho = 1/(dmd_pixel*magnification)
        self.f_l = 1/(deg_inc*np.pi/180*r_i)
        self.anisotropy_factor = self.f_rho/self.f_l
        self.n_l = int(self.f_l*l_max)
        self.n_rho = int(self.f_rho*(self.r_o - self.r_i))
        self.n_z = int(self.f_rho*tau)

    @property
    def shape(self):
        return (self.n_rho, self.n_l, self.n_z)


def _implicit(kind, cx, cy, cz, sx, sy, sz, c2x, c2y, c2z, T):
    #Implicit surfaces of vamtoolbox devR2R buildtargetsR2R.tpms (same expressions, same evaluation order)
    if kind == "gyroid-sheet":
        return np.where((cx*sy + cy*sz + cz*sx)**2 - T**2 <= 0, 1, 0)
    if kind == "gyroid-skel":
        return np.where(cx*sy + cy*sz + cz*sx - T <= 0, 1, 0)
    if kind == "schwarzp-skel":
        return np.where(np.abs(cx + cy + cz - T) <= 0.4, 1, 0)
    if kind == "iwp-skel":
        return np.where(2*(cx*cy + cy*cz + cz*cx) - (c2x + c2y + c2z) + T <= 0, 1, 0)
    if kind == "diamond-skel":
        return np.where((cx*cy*cz) - (sx*sy*sz) + T <= 0, 1, 0)
    if kind == "diamond-sheet":
        return np.where(((cx*cy*cz) - (sx*sy*sz))**2 - T**2 <= 0, 1, 0)
    raise ValueError(f"Unsupported TPMS type '{kind}'.")


def generateTPMSTarget(sampling=None, unit_cell_size=None, t=(0.5,), kind="diamond-skel"):
    """
    TPMS lattice filling the whole band (N_rho, N_l, N_z) of `sampling`, as float64 0/1.

    Parameters
    ----------
    sampling : R2RSampling
        Default R2RSampling() = reference case "tpms_diamond_skel0" (r_i 0.76, tau 0.3, l_max 5, deg_inc 0.5 -> 208 x 754 x 208).
    unit_cell_size : float
        Lattice period (cm). The reference uses the web thickness tau (default).
    t : sequence of float
        Level-set offset. One value: constant. Two values: linear gradient along l (reference behaviour).
    kind : str
        'diamond-skel' (reference default), 'diamond-sheet', 'gyroid-skel', 'gyroid-sheet', 'schwarzp-skel', 'iwp-skel'.
    """
    sampling = R2RSampling() if sampling is None else sampling
    unit_cell_size = sampling.tau if unit_cell_size is None else unit_cell_size
    shape = sampling.shape

    x_dim = shape[0]/sampling.f_rho
    y_dim = shape[1]/sampling.f_l
    z_dim = shape[2]/sampling.f_rho
    _x = np.linspace(0, x_dim, shape[0])
    _y = np.linspace(0, y_dim, shape[1])
    _z = np.linspace(-z_dim/2, z_dim/2, shape[2])

    unit_cell_pitch = 1/unit_cell_size
    k = 2*np.pi*unit_cell_pitch

    #1D factors broadcast to (N_rho, N_l): same values as the reference's full meshgrid
    cx, sx, c2x = np.cos(k*_x)[:, None], np.sin(k*_x)[:, None], np.cos(2*k*_x)[:, None]
    cy, sy, c2y = np.cos(k*_y)[None, :], np.sin(k*_y)[None, :], np.cos(2*k*_y)[None, :]
    t = list(t)
    if len(t) == 2:
        T = np.broadcast_to(np.linspace(t[0], t[1], shape[1])[None, :], shape[:2])
    else:
        T = np.ones(shape[:2])*t

    cz_all, sz_all, c2z_all = np.cos(k*_z), np.sin(k*_z), np.cos(2*k*_z)

    target = np.empty(shape)
    for i_z in range(shape[2]):
        cz, sz, c2z = cz_all[i_z], sz_all[i_z], c2z_all[i_z]
        target[:, :, i_z] = _implicit(kind, cx, cy, cz, sx, sy, sz, c2x, c2y, c2z, T)
    return target
