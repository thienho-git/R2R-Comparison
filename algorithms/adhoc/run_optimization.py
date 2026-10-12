import vamtoolbox

from .r2r_wrapping import minimizeR2RBCLP


def bclpOptions(**kwargs):
    """
    vamtoolbox Options for BCLP, completed with the options read by the local BCLP.py that vamtoolbox may not define.
    kwargs are passed to vamtoolbox.optimize.Options (method, n_iter, p, q, eps, weight, learning_rate, g0, ...).
    """
    options = vamtoolbox.optimize.Options(**{"method": "BCLP", **kwargs})

    # Options used by the local BCLP.py that vamtoolbox may not define by default
    defaults = {
        "momentum": 0.0,
        "regularize_l": 0.0,
        "g0": None,
        "blb": 0,
        "bub": None,
        "bit_depth": None,
        "exit_param": None,
        "save_img_path": None,
    }
    for key, value in defaults.items():
        if not hasattr(options, key):
            setattr(options, key, value)
    return options


def runOptimization(target_geo, propagator, n_iter=50, learning_rate=3e-6, eps=0.1, p=2, q=1, verbose="time"):
    """
    Run the band constraint Lp norm optimization (local BCLP.py) in the UW domain with the R2R propagator.
    The step size depends on the propagation matrix scale. With the vamtoolbox standard projector (default),
    1e-6 decreases the loss monotonically and 3e-6 is faster but slightly noisy; 1e-5 and above oscillate.
    With the precomputed data/ matrix, use ~1e-4.

    Returns
    -------
    opt_sino : ndarray (N_r, N_theta, N_z)
    dose : ndarray (N_rho, N_long, N_z)
        dose (back propagation of the optimized sinogram) in the UW domain
    mapped_dose : ndarray (N_rho, N_long, N_z)
        material response to the dose
    logs : BCLP.LogPerf
    """
    options = bclpOptions(
        n_iter=n_iter,
        filter=None, #unused: the R2R propagator provides inverseBackward() for the initial sinogram
        response_model=vamtoolbox.response.ResponseModel(),
        eps=eps,
        weight=1,
        p=p,
        q=q,
        learning_rate=learning_rate,
        optim_alg="grad_des",
        verbose=verbose,
    )
    return minimizeR2RBCLP(target_geo.array, propagator, options)
