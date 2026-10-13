"""
R2R propagation in the unwrapped (UW) domain and BCLP optimization, following the R2R article
(R2R_tomographic_volumetric_additive_manufacturing_article_UCB.pdf: Figure 2, "Numerical light propagation model", S2, S11).
The wrapping T: UW -> W is already inside the propagation matrix, so nothing here wraps.

Domains (article, Fig. 2A):
    W  (wrapped)  : photoresist curved on the cylindrical exposure roller, x-y coordinates, origin on the roller axis.
                    Physical light propagation happens here.
    UW (unwrapped): the same photoresist as a flat web, rho-l coordinates (rho = thickness, l = length along the web).
                    The target (prescription dose) and the optimized dose are defined here.

Algorithm (article, Fig. 2B/2C and S2):
    T: UW -> W is folded into a composite, shift-invariant, single-angle propagation matrix
        P_UW->g = T_W->UW(P_W->g),   shape N_r x (N_rho*N_l)
    The long UW tomogram is cut into blocks f_theta_1 ... f_theta_N by a shifting window ("unfold"), one block per
    angle/time step, because the web moves by one l-voxel per angular increment.
    Forward : [g_theta_1 ... g_theta_N] = P_UW->g @ [f_theta_1 ... f_theta_N]   (matrix-matrix product = sinogram)
    Backward: [f+_theta_1 ... f+_theta_N] = P*_UW->g @ [g_theta_1 ... g_theta_N], then accumulated ("fold") into the
              long UW tomogram.
    z-slices are independent, so the same matrix is reused for every z.

Array conventions used in this file:
    UW tomogram : (N_rho, N_long, N_z), C order. Within one window, column index of P is i_rho*N_l + i_l.
    Sinogram    : (N_r, N_theta, N_z) with N_theta = N_long + N_l - 1, i.e. (detector, angle, z) like vamtoolbox,
                  so BCLP.py's per-angle discretization (axis 1) still applies.

Conventions of the precomputed matrices (matrix_sino2uw_*.pt, built by vamtoolbox devR2R pyTorchRayTraceR2R; geometry
in devR2R projector/transformationR2R.WrapTransformation, see algorithms/reference/original/vamtoolbox_devR2R):
    - shape 1475 x 149552 = N_r x (N_rho*N_l) with N_rho = int(tau*f_rho) = 208, N_l = int(2*pi*r_i*f_l) = 719.
    - rho grid linspace(r_i, r_o, N_rho): row 0 is the INNER face r = r_i. The central detector row deposits most
      energy at i_rho ~ 1 and decreases with i_rho, i.e. light enters the web through the inner face.
    - l pitch r_i*deg_inc (f_l = 1/(deg_inc*r_i)): the window is a quarter-turn straight run (r_i*pi/2 = 1.19 cm) on
      +x, the 180 deg wrap around the bottom of the roller, and a quarter-turn straight run on -x.
    - detector row 0 sees l ~ 521-606 and row 1474 sees l ~ 112-197, so l = 0 is on the +x straight run when the
      detector coordinate x increases with row index.
Remaining assumption: block i = f[:, i:i+N_l] for time step i (direction of web travel).
"""

import types

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F

from . import BCLP


# ----------------------------------------------------------------------------------------------------------------------
# Loading the composite propagation matrix P_UW->g
# ----------------------------------------------------------------------------------------------------------------------

def loadPropagationMatrix(path):
    """
    Load P_UW->g (N_r x N_rho*N_l) as a float64 scipy CSR matrix from a .pt file (torch sparse COO), read with
    torch.load(weights_only=True), which does not execute arbitrary code.
    """
    tensor = torch.load(path, map_location="cpu", weights_only=True).coalesce()
    indices, values, shape = tensor.indices().numpy(), tensor.values().numpy(), tuple(tensor.shape)
    return scipy.sparse.coo_matrix((values.astype(np.float64), (indices[0], indices[1])), shape=shape).tocsr()


def inferWindowShape(P):
    """
    Recover (N_rho, N_l) of the UW window from the matrix structure.
    The central detector ray crosses the web thickness at a single l, so consecutive non-zero columns on that row
    differ by exactly N_l (column index = i_rho*N_l + i_l).
    """
    row = P.getrow(P.shape[0]//2)
    strides = np.diff(np.sort(row.indices))
    values, counts = np.unique(strides[strides > 1], return_counts=True)
    n_l = int(values[np.argmax(counts)])
    if P.shape[1] % n_l != 0:
        raise ValueError(f"Could not infer the window shape: {P.shape[1]} columns is not a multiple of N_l = {n_l}.")
    return P.shape[1]//n_l, n_l


# ----------------------------------------------------------------------------------------------------------------------
# Shift-window propagator (unfold -> P_UW->g -> sinogram ; sinogram -> P*_UW->g -> fold)
# ----------------------------------------------------------------------------------------------------------------------

def _toTorchCSR(P, device, dtype):
    P = scipy.sparse.csr_matrix(P)
    return torch.sparse_csr_tensor(torch.from_numpy(P.indptr.astype(np.int64)), torch.from_numpy(P.indices.astype(np.int64)),
                                   torch.from_numpy(P.data), size=P.shape, dtype=dtype, device=device)


class R2RAlgebraicPropagator:
    """
    Projector with the forward()/backward() interface expected by BCLP.py, but operating on a long UW tomogram.

    forward(f) : f (N_rho, N_long[, N_z]) -> g (N_r, N_theta, N_z)
    backward(g): g (N_r, N_theta, N_z) or flattened -> f (N_rho, N_long, N_z)
    backward is the exact adjoint of forward (checked by adjointTest()).

    The tomogram is the bare target: forward pads it by N_l - 1 zeros on both ends along l (every voxel travels
    through the whole window) and backward crops the dose back to the target, so N_theta = N_long + N_l - 1. This is
    what the reference (vamtoolbox devR2R PyTorchAlgebraicR2RPropagator, torch.nn.Unfold/Fold with padding N_l - 1)
    does: the voxels outside the target are not part of the optimization.

    The products with P and P* run in torch (sparse CSR) on `device` (default: CUDA when available, else CPU), z_batch
    z slices at a time, so large 3D sinograms stay in host memory. Inputs and outputs are numpy arrays of `dtype`
    (float64 by default; float32 halves the memory and matches the reference), so BCLP.py is unchanged.

    Time step i sees the window f[:, i:i+N_l] (block f_theta_i) of the padded tomogram. chunk = number of blocks
    unfolded at once (device memory ~ itemsize*N_rho*N_l*chunk*z_batch bytes).
    """

    def __init__(self, P, n_rho, n_l, chunk=128, z_batch=4, device=None, dtype=torch.float64):
        if P.shape[1] != n_rho*n_l:
            raise ValueError(f"Matrix has {P.shape[1]} columns but N_rho*N_l = {n_rho*n_l}.")
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        P = scipy.sparse.csr_matrix(P, dtype=np.float64)
        self.P = _toTorchCSR(P, self.device, dtype)
        self.P_H = _toTorchCSR(P.T, self.device, dtype) #P*_UW->g (adjoint) for back propagation
        self.n_r = P.shape[0]
        self.n_rho = n_rho
        self.n_l = n_l
        self.chunk = chunk
        self.z_batch = z_batch
        self.pad = n_l - 1

    @classmethod
    def fromMatrixFile(cls, path, **kwargs):
        """Propagator of a precomputed P_UW->g (.pt), window N_rho x N_l read from the matrix structure."""
        P = loadPropagationMatrix(path)
        return cls(P, *inferWindowShape(P), **kwargs)

    def nAngles(self, n_long):
        return n_long + 2*self.pad - self.n_l + 1

    def _toDevice(self, array):
        return torch.as_tensor(np.ascontiguousarray(array, dtype=self.np_dtype)).to(self.device)

    @torch.no_grad()
    def forward(self, f):
        f = np.asarray(f)
        if f.ndim == 2:
            f = f[:, :, None]
        n_long, n_z = f.shape[1], f.shape[2]
        n_theta = self.nAngles(n_long)
        if f.shape[0] != self.n_rho or n_theta < 1:
            raise ValueError(f"UW tomogram must be (N_rho={self.n_rho}, N_long>={self.n_l}, N_z), got {f.shape}.")

        g = np.empty((self.n_r, n_theta, n_z), dtype=self.np_dtype)
        for z0 in range(0, n_z, self.z_batch):
            z1 = min(z0 + self.z_batch, n_z)
            f_z = F.pad(self._toDevice(f[:, :, z0:z1]), (0, 0, self.pad, self.pad))
            #unfold: windows[i_rho, i, z, j] = f_z[i_rho, i + j, z] (a view, no copy)
            windows = f_z.unfold(1, self.n_l, 1)
            g_z = torch.empty((self.n_r, n_theta, z1 - z0), device=self.device, dtype=self.dtype)
            for t0 in range(0, n_theta, self.chunk):
                t1 = min(t0 + self.chunk, n_theta)
                #blocks [f_theta_t0 ... f_theta_t1-1] as columns, one column per (angle, z): (N_rho*N_l, c*n_zb)
                blocks = windows[:, t0:t1].permute(0, 3, 1, 2).reshape(self.n_rho*self.n_l, -1)
                g_z[:, t0:t1, :] = (self.P @ blocks).reshape(self.n_r, t1 - t0, z1 - z0)
            g[:, :, z0:z1] = g_z.cpu().numpy()
        return g

    @torch.no_grad()
    def backward(self, g):
        g = np.asarray(g)
        if g.ndim == 1:
            g = g.reshape(self.n_r, -1, 1) #flattened input is assumed to be a single z-slice
        elif g.ndim == 2:
            g = g[:, :, None]
        n_theta, n_z = g.shape[1], g.shape[2]
        n_long = n_theta + self.n_l - 1 #padded length

        f = np.empty((self.n_rho, n_long - 2*self.pad, n_z), dtype=self.np_dtype)
        for z0 in range(0, n_z, self.z_batch):
            z1 = min(z0 + self.z_batch, n_z)
            n_zb = z1 - z0
            g_z = self._toDevice(g[:, :, z0:z1])
            f_z = torch.zeros((self.n_rho, n_long, n_zb), device=self.device, dtype=self.dtype)
            for t0 in range(0, n_theta, self.chunk):
                t1 = min(t0 + self.chunk, n_theta)
                c = t1 - t0
                #back propagate every block of the chunk at once: (N_rho, N_l, c, n_zb)
                blocks = (self.P_H @ g_z[:, t0:t1, :].reshape(self.n_r, -1)).reshape(self.n_rho, self.n_l, c, n_zb)
                #fold: accumulate block f+_theta_i into the long tomogram at the window position of time step i
                #(F.fold with a 1 x N_l kernel sums the overlapping windows; channels = (rho, z))
                cols = blocks.permute(0, 3, 1, 2).reshape(1, self.n_rho*n_zb*self.n_l, c)
                folded = F.fold(cols, output_size=(1, c + self.n_l - 1), kernel_size=(1, self.n_l))
                f_z[:, t0:t1 + self.n_l - 1, :] += folded.reshape(self.n_rho, n_zb, -1).permute(0, 2, 1)
            f[:, :, z0:z1] = f_z[:, self.pad:n_long - self.pad].cpu().numpy()
        return f

    def adjointTest(self, n_long=None, n_z=2, seed=0):
        #<P f, g> must equal <f, P* g> for random f, g. Returns the relative difference.
        rng = np.random.default_rng(seed)
        n_long = self.n_l + 20 if n_long is None else n_long
        f = rng.random((self.n_rho, n_long, n_z))
        g = rng.random((self.n_r, self.nAngles(n_long), n_z))
        lhs, rhs = np.sum(self.forward(f)*g, dtype=np.float64), np.sum(f*self.backward(g), dtype=np.float64)
        return abs(lhs - rhs)/abs(lhs)


# ----------------------------------------------------------------------------------------------------------------------
# UW target and BCLP optimization in the UW domain
# ----------------------------------------------------------------------------------------------------------------------

class UWTargetGeometry:
    """Minimal stand-in for vamtoolbox.geometry.TargetGeometry: BCLP.py only needs .array, here (N_rho, N_long, N_z)."""

    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float64)


def minimizeR2RBCLP(target_uw, propagator, options):
    """
    Run BCLP.py's band constraint Lp-norm optimization with the target and dose in the UW domain.
    `propagator` (built from the ray-traced matrix) is passed to BCLP.BCLPNorm as its projector self.P.

    Returns
    -------
    g_opt : (N_r, N_theta, N_z) optimized sinogram (detector x angle/time x z)
    dose_uw, mapped_dose_uw : (N_rho, N_long, N_z) dose and material response in the UW domain
    logs : BCLP.LogPerf
    """
    target_geo = UWTargetGeometry(target_uw)
    n_theta = propagator.nAngles(np.atleast_3d(target_geo.array).shape[1])
    proj_geo = types.SimpleNamespace(n_angles=n_theta, angles=np.arange(n_theta))

    bclp = BCLP.BCLPNorm(target_geo, proj_geo, options, projector=propagator)
    g_opt = bclp.checkSinogramShape(bclp.gradientDescent(), desired_shape="cylindrical")
    return g_opt, bclp.dose, bclp.mapped_dose, bclp.logs
