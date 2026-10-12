"""
Rectangular prism inside the photoresist web, directly in the unwrapped (UW) domain: a box inside a band of finite
width (the web thickness), not cropped by a cylinder. It is wrapped onto the roller later by the propagation
(algorithms/adhoc/r2r_wrapping.py).

Rows are rho (through the web thickness, row 0 = rho 0 = light-entry face) and columns are l (along the web). The
prism is given as one 2D slice (one z position across the web width).
"""

import numpy as np
from PIL import Image


def _inside(n, size, center):
    #Voxel centers normalized so the band spans [-0.5, 0.5]
    u = (np.arange(n) + 0.5)/n - 0.5
    return np.abs(u - center) <= size/2


def generatePrismSlice(filename="prism_slice.png", thickness_px=208, length_px=1000, width=1, height=0.5, center=(0.0, 0.25)):
    """
    Binary image of a cross-section (z slice) of a rectangular prism inside the web.
    The band covers the whole image; its physical size is set in createGeometry (thickness tau from the matrix, band_length).

    Parameters
    ----------
    filename : str
        Output .png path.
    thickness_px, length_px : int
        Image size across the band (rho) and along the band (l), in pixels. Resampled to the voxel grid in createGeometry.
    width, height : float
        Rectangle size as a fraction of the band length (width, along l) and band thickness (height, along rho), 0 to 1.
    center : tuple of float
        Rectangle center offset (l, rho) from the band center, as a fraction of the band length and thickness.
        Positive rho offset moves away from the light-entry face.

    Returns
    -------
    filename : str
        Path of the saved image, to be passed to createGeometry.
    """
    image = (generatePrismImage((thickness_px, length_px), width, height, center)*255).astype(np.uint8)

    Image.fromarray(image, mode="L").save(filename)

    return filename


def generatePrismImage(shape, width=1, height=0.5, center=(0.0, 0.25)):
    """
    2D image (N_rho, N_l) of a z slice of a rectangular prism, float64 0/1: a rectangle inside the band.

    width, height : float
        Rectangle size as a fraction of the band along l (width) and rho (height), 0 to 1.
    center : tuple of float
        Rectangle center offset (l, rho) from the band center, as a fraction of the band along each axis.
    """
    n_rho, n_l = shape
    inside = _inside(n_rho, height, center[1])[:, None] & _inside(n_l, width, center[0])[None, :]
    return inside.astype(np.float64)


if __name__ == "__main__":
    print("Saved", generatePrismSlice())
