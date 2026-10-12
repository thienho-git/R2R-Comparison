"""
Wrapped (W) / unwrapped (UW) part of roll-to-roll (R2R) CAL, following the R2R article
(R2R_tomographic_volumetric_additive_manufacturing_article_UCB.pdf: Figure 2, "Numerical light propagation model", S2, S11).

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
    Sinogram    : (N_r, N_theta, N_z) with N_theta = N_long - N_l + 1, i.e. (detector, angle, z) like vamtoolbox,
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
Remaining assumptions (configurable): block i = f[:, i:i+N_l] for time step i (direction of web travel), and the
initial-guess scaling in R2RAlgebraicPropagator.inverseBackward.
"""

import os
import re
import types

import numpy as np
import scipy.ndimage
import scipy.sparse
import torch
import torch.nn.functional as F

from . import BCLP


# ----------------------------------------------------------------------------------------------------------------------
# Loading the composite propagation matrix P_UW->g
# ----------------------------------------------------------------------------------------------------------------------

def parseMatrixFilename(path):
    """
    Read the physical parameters encoded in the matrix file name, e.g.
    'matrix_sino2uw_alpha_mat=2.2_n_sur=1.33_n_mat=1.49_r_i=0.76_tau=0.3_deg_inc=0.5.pt'
    -> {'alpha_mat': 2.2, 'n_sur': 1.33, 'n_mat': 1.49, 'r_i': 0.76, 'tau': 0.3, 'deg_inc': 0.5}
    Lengths are in cm (article: r_i, tau, alpha in cm^-1), deg_inc in degrees.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return {key: float(value) for key, value in re.findall(r"(?:^|_)([A-Za-z]+(?:_[A-Za-z]+)*)=([-+0-9.eE]+)(?=_|$)", stem)}


def loadPropagationMatrix(path):
    """
    Load P_UW->g (N_r x N_rho*N_l) as a float64 scipy CSR matrix from .pt (torch sparse COO) or .npz (scipy sparse).
    The .pt file is read with torch.load(weights_only=True), which does not execute arbitrary code.
    """
    if path.lower().endswith(".npz"):
        return scipy.sparse.load_npz(path).tocsr().astype(np.float64)

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
# Geometry of the transformation T between UW (rho, l) and W (x, y)
# ----------------------------------------------------------------------------------------------------------------------

class R2RGeometry:
    """
    Geometry of one UW window and its wrapped (W) shape on the roller (article Fig. 2A, S11).

    Layout in W (origin on the roller axis, y along the beam), as in the reference (devR2R WrapTransformation):
        - the window centre l_center lies on the bottom of the roller, where the central ray crosses the web radially;
        - the wrap covers +-90 deg around it (N_curve = 180/deg_inc voxels along l);
        - outside the wrap the web runs straight along +y on both sides (x = +r for small l, x = -r for large l).
    Radius of a voxel: r = r_i + (i_rho + 0.5)*d_rho when rho0_face == 'inner' (reference rho grid starts at r_i, and
    the matrix shows light entering through that face, i.e. travelling towards -y through the roller; default), or
    r = r_o - (i_rho + 0.5)*d_rho when rho0_face == 'outer'.
    The l pitch d_l = r_ref*deg_inc with r_ref = r_i (reference f_l = 1/(deg_inc*r_i)); pass r_ref = r_i + tau/2 for
    a neutral axis at mid-thickness.
    All lengths in the units of r_i and tau (cm for the provided matrix).
    """

    def __init__(self, r_i, tau, deg_inc, n_rho, n_l, n_r=None, rho0_face="inner", l_center=None, r_ref=None):
        self.r_i = r_i
        self.tau = tau
        self.r_o = r_i + tau
        self.dtheta = np.deg2rad(deg_inc)
        self.n_rho = n_rho
        self.n_l = n_l
        self.n_r = n_r
        self.rho0_face = rho0_face
        self.d_rho = tau/n_rho
        self.r_ref = r_i if r_ref is None else r_ref
        self.d_l = self.r_ref*self.dtheta
        self.l_center = (n_l - 1)/2 if l_center is None else l_center
        self.n_quarter = (np.pi/2)/self.dtheta #number of l voxels from window centre to each tangent point
        self.l_tangent_plus_x = self.l_center - self.n_quarter #wrap ends here and the +x straight run starts (smaller l)
        self.l_tangent_minus_x = self.l_center + self.n_quarter #wrap ends here and the -x straight run starts (larger l)
        self.detector_pitch = None if n_r is None else 2*self.r_o/n_r

    @classmethod
    def fromMatrixFile(cls, path, P=None, **kwargs):
        #Build the geometry from the matrix file name (r_i, tau, deg_inc) and the matrix structure (N_rho, N_l, N_r).
        params = parseMatrixFilename(path)
        P = loadPropagationMatrix(path) if P is None else P
        n_rho, n_l = inferWindowShape(P)
        return cls(params["r_i"], params["tau"], params["deg_inc"], n_rho, n_l, n_r=P.shape[0], **kwargs)

    @classmethod
    def fromStraightLength(cls, r_i, tau, deg_inc, n_rho, straight_length=None, **kwargs):
        """
        Build the window from physical parameters only (no matrix file): the 180 deg wrap (180/deg_inc voxels + 1)
        plus a straight run of straight_length on each side, with d_l = r_ref*deg_inc. Default straight_length is the
        reference quarter turn r_i*pi/2 (1.19 cm for r_i = 0.76 cm; the article quotes L = 1.41 cm).
        """
        straight_length = r_i*np.pi/2 if straight_length is None else straight_length
        d_l = (kwargs.get("r_ref") or r_i)*np.deg2rad(deg_inc)
        n_wrap = 2*int(round(90/deg_inc)) + 1
        n_l = n_wrap + 2*int(np.ceil(straight_length/d_l - 1e-9))
        return cls(r_i, tau, deg_inc, n_rho, n_l, **kwargs)

    def _radius(self, i_rho):
        if self.rho0_face == "outer":
            return self.r_o - (i_rho + 0.5)*self.d_rho
        return self.r_i + (i_rho + 0.5)*self.d_rho

    def _rhoIndex(self, r):
        if self.rho0_face == "outer":
            return (self.r_o - r)/self.d_rho - 0.5
        return (r - self.r_i)/self.d_rho - 0.5

    def uwToXY(self, i_rho, i_l):
        """T: UW -> W for (fractional) voxel indices. Returns x, y."""
        i_rho, i_l = np.broadcast_arrays(np.asarray(i_rho, float), np.asarray(i_l, float))
        r = self._radius(i_rho)
        phi = (self.l_center - i_l)*self.dtheta #0 at the bottom of the roller, positive towards +x
        x = r*np.sin(phi)
        y = -r*np.cos(phi)

        straight_plus = i_l < self.l_tangent_plus_x
        x = np.where(straight_plus, r, x)
        y = np.where(straight_plus, (self.l_tangent_plus_x - i_l)*self.d_l, y)

        straight_minus = i_l > self.l_tangent_minus_x
        x = np.where(straight_minus, -r, x)
        y = np.where(straight_minus, (i_l - self.l_tangent_minus_x)*self.d_l, y)
        return x, y

    def xyToUW(self, x, y):
        """T: W -> UW. Returns fractional (i_rho, i_l) and a mask of points that lie inside the window's web."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        on_wrap = y <= 0
        r = np.where(on_wrap, np.hypot(x, y), np.abs(x))
        i_l_wrap = self.l_center - np.arctan2(x, -y)/self.dtheta
        i_l_straight = np.where(x >= 0, self.l_tangent_plus_x - y/self.d_l, self.l_tangent_minus_x + y/self.d_l)
        i_l = np.where(on_wrap, i_l_wrap, i_l_straight)
        i_rho = self._rhoIndex(r)
        inside = (r >= self.r_i) & (r <= self.r_o) & (i_l >= -0.5) & (i_l <= self.n_l - 0.5)
        return i_rho, i_l, inside

    def wGrid(self, pixel=None):
        """
        Cartesian W grid that contains the whole window: x in [-r_o, r_o], y from -r_o to the top of the straight runs.
        Returns x (n_x,), y (n_y,) pixel-centre coordinates. Default pixel = d_rho.
        """
        pixel = self.d_rho if pixel is None else pixel
        y_top = (self.l_tangent_plus_x + 0.5)*self.d_l
        x = np.arange(-self.r_o + pixel/2, self.r_o, pixel)
        y = np.arange(-self.r_o + pixel/2, y_top, pixel)
        return x, y

    def uwToW(self, f_uw, pixel=None, fill_value=np.nan):
        """
        Resample one UW window f_uw (N_rho, N_l) onto the W grid (bilinear). Returns image (n_y, n_x), x, y.
        Pixels outside the web are fill_value. Image rows follow y, columns follow x (use origin='lower' to plot).
        """
        x, y = self.wGrid(pixel)
        X, Y = np.meshgrid(x, y)
        i_rho, i_l, inside = self.xyToUW(X, Y)
        image = scipy.ndimage.map_coordinates(np.asarray(f_uw, float), [i_rho, i_l], order=1, mode="nearest")
        return np.where(inside, image, fill_value), x, y

    def wToUW(self, f_w, x, y):
        """Resample a W image f_w (n_y, n_x) on grid (x, y) into one UW window (N_rho, N_l) (bilinear)."""
        i_rho, i_l = np.meshgrid(np.arange(self.n_rho), np.arange(self.n_l), indexing="ij")
        xs, ys = self.uwToXY(i_rho, i_l)
        col = (xs - x[0])/(x[1] - x[0])
        row = (ys - y[0])/(y[1] - y[0])
        return scipy.ndimage.map_coordinates(np.asarray(f_w, float), [row, col], order=1, mode="constant", cval=0.0)

    def uwToWMatrix(self, x, y):
        """
        Sparse matrix T (n_y*n_x) x (N_rho*N_l) such that vec(f_W) = T @ vec(f_UW) (bilinear, C order).
        This is the operator that is subsumed into the composite matrix: P_UW->g = P_W->g @ T (see composeUWMatrix).
        """
        X, Y = np.meshgrid(x, y)
        i_rho, i_l, inside = self.xyToUW(X.ravel(), Y.ravel())
        rows = np.flatnonzero(inside)
        i_rho = np.clip(i_rho[rows], 0, self.n_rho - 1)
        i_l = np.clip(i_l[rows], 0, self.n_l - 1)
        r0 = np.minimum(np.floor(i_rho).astype(int), self.n_rho - 2)
        l0 = np.minimum(np.floor(i_l).astype(int), self.n_l - 2)
        a, b = i_rho - r0, i_l - l0
        corners = [(0, 0, (1 - a)*(1 - b)), (1, 0, a*(1 - b)), (0, 1, (1 - a)*b), (1, 1, a*b)]
        data = np.concatenate([w for _, _, w in corners])
        cols = np.concatenate([(r0 + dr)*self.n_l + (l0 + dl) for dr, dl, _ in corners])
        return scipy.sparse.csr_matrix((data, (np.tile(rows, 4), cols)), shape=(X.size, self.n_rho*self.n_l))


def composeUWMatrix(P_w, T_uw_to_w):
    """
    Article: P_W->g f_W = g  and  T_W->UW(P_W->g) f_UW = g  ->  P_UW->g = P_W->g @ T.
    Use this when a single-angle W-domain matrix (e.g. from ray tracing on the x-y grid of geometry.wGrid()) is
    available instead of the precomputed composite matrix.
    """
    return (scipy.sparse.csr_matrix(P_w) @ T_uw_to_w).tocsr()


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

    N_theta = N_long - N_l + 1 when the tomogram is already zero-padded (padTarget, default). With
    implicit_padding=True the tomogram is the bare target: forward pads it by N_l - 1 zeros on both ends along l and
    backward crops the dose back to the target, so N_theta = N_long + N_l - 1. This is what the reference
    (vamtoolbox devR2R PyTorchAlgebraicR2RPropagator, torch.nn.Unfold/Fold with padding N_l - 1) does: the voxels
    outside the target are then not part of the optimization.

    The products with P and P* run in torch (sparse CSR) on `device` (default: CUDA when available, else CPU), z_batch
    z slices at a time, so large 3D sinograms stay in host memory. Inputs and outputs are numpy arrays of `dtype`
    (float64 by default; float32 halves the memory and matches the reference), so BCLP.py is unchanged.

    Time step i sees the window f[:, i:i+N_l] (block f_theta_i) of the padded tomogram. reverse_time=True flips the
    angle axis, i.e. the web travels the other way. chunk = number of blocks unfolded at once
    (device memory ~ itemsize*N_rho*N_l*chunk*z_batch bytes).
    """

    def __init__(self, P, n_rho, n_l, reverse_time=False, chunk=128, z_batch=4, device=None, dtype=torch.float64,
                 implicit_padding=False):
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
        self.reverse_time = reverse_time
        self.chunk = chunk
        self.z_batch = z_batch
        self.implicit_padding = implicit_padding
        self.pad = n_l - 1 if implicit_padding else 0

    def nAngles(self, n_long):
        return n_long + 2*self.pad - self.n_l + 1

    def _toDevice(self, array):
        return torch.as_tensor(np.ascontiguousarray(array, dtype=self.np_dtype)).to(self.device)

    def _toNumpy(self, tensor):
        return tensor.cpu().numpy()

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
            g[:, :, z0:z1] = self._toNumpy(g_z.flip(1) if self.reverse_time else g_z)
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
            if self.reverse_time:
                g_z = g_z.flip(1)
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
            f[:, :, z0:z1] = self._toNumpy(f_z[:, self.pad:n_long - self.pad])
        return f

    def inverseBackward(self, f, filter=None):
        """
        Initial sinogram for BCLP.py (it calls this when available). Not from the article: forward projection of the
        target, scaled so that its back projection best matches the target in the least-squares sense.
        filter is accepted for interface compatibility and ignored (ramp filtering assumes conventional CAL).
        """
        g = self.forward(f)
        b = self.backward(g)
        f = np.atleast_3d(np.asarray(f, float))
        scale = np.sum(f*b)/max(np.sum(b*b), np.finfo(float).tiny)
        return scale*g

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


def padTarget(f_uw, n_l):
    """
    Zero-pad a UW target (N_rho, N_obj[, N_z]) by N_l - 1 voxels on both ends along l, so that every voxel of the
    object travels through the whole window (enters before the first angle, leaves after the last).
    Resulting number of angles: N_theta = N_obj + N_l - 1.
    """
    f_uw = np.asarray(f_uw, dtype=np.float64)
    pad = [(0, 0)]*f_uw.ndim
    pad[1] = (n_l - 1, n_l - 1)
    return np.pad(f_uw, pad)


def imageToUWTarget(image_file, n_rho, n_obj):
    """
    Load a grayscale image as a UW target: image rows -> rho (thickness, row 0 = rho 0 = light-entry face),
    image columns -> l (along the web). Resized to (n_rho, n_obj) and normalized to [0, 1].
    """
    from PIL import Image
    image = Image.open(image_file).convert("L").resize((n_obj, n_rho), Image.NEAREST)
    array = np.asarray(image, dtype=np.float64)
    return array/array.max() if array.max() > 0 else array


def minimizeR2RBCLP(target_uw, propagator, options):
    """
    Run BCLP.py's band constraint Lp-norm optimization with the target and dose in the UW domain.
    `propagator` (e.g. built from the ray-traced data/ matrix) is passed to BCLP.BCLPNorm as its projector self.P.

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
