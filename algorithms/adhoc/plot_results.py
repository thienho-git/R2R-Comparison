import numpy as np
import matplotlib.pyplot as plt


def plotResults(target_geo, r2r_geo, opt_sino, mapped_dose, logs, time_index=None):
    """
    Plot the target and mapped dose in the unwrapped (UW) domain, the optimized sinogram, the same target and mapped dose
    wrapped (W) onto the roller at one time step, then the convergence metrics.

    UW panels: flat web, horizontal = l (along the web), vertical = rho (rho = 0, the light-entry face, on top).
    W panels: the window seen at time_index wrapped with T: UW -> W (rho = 0 on the inner face r_i, where the light
    enters; gray = outside the web).
    time_index is a roller position (sinogram column / angle), not an optimization iteration: the dose shown is the one of
    the final iterate, and time_index only picks which part of the web sits on the roller. The default (middle column)
    centers the band on the bottom of the roller.
    """
    target = np.atleast_3d(target_geo.array)[:, :, 0]
    mapped = np.atleast_3d(mapped_dose)[:, :, 0]
    sino = np.atleast_3d(opt_sino)[:, :, 0]
    time_index = (sino.shape[1] - 1)//2 if time_index is None else time_index
    window = slice(time_index, time_index + r2r_geo.n_l)
    mm = 10 #matrix units are cm
    uw_extent = [0, target.shape[1]*r2r_geo.d_l*mm, r2r_geo.tau*mm, 0]

    cmap = plt.get_cmap("CMRmap").copy()
    cmap.set_bad("lightgray")
    gray = plt.get_cmap("gray").copy()
    gray.set_bad("lightgray")

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, array, title, cm, vmax in [(axs[0, 0], target, "Target (UW, flat web)", gray, 1),
                                       (axs[0, 1], mapped, "Mapped dose (UW, flat web)", cmap, None)]:
        im = ax.imshow(array, cmap=cm, vmin=0, vmax=vmax, aspect="auto", extent=uw_extent, interpolation="none")
        ax.axvspan(window.start*r2r_geo.d_l*mm, window.stop*r2r_geo.d_l*mm, fill=False, edgecolor="c", linestyle="--")
        ax.set(title=title, xlabel="l along web (mm)", ylabel="rho through thickness (mm)")
        fig.colorbar(im, ax=ax)

    im = axs[0, 2].imshow(sino, cmap="CMRmap", aspect="auto", interpolation="none")
    axs[0, 2].axvline(time_index, color="c", linestyle="--")
    axs[0, 2].set(title="Optimized sinogram", xlabel="angle / time step", ylabel="detector r")
    fig.colorbar(im, ax=axs[0, 2])

    for ax, array, title, cm, vmax in [(axs[1, 0], target, "Target", gray, 1),
                                       (axs[1, 1], mapped, "Mapped dose", cmap, None)]:
        image, x, y = r2r_geo.uwToW(array[:, window])
        im = ax.imshow(np.ma.masked_invalid(image), cmap=cm, vmin=0, vmax=vmax, origin="lower", interpolation="none",
                       extent=[x[0]*mm, x[-1]*mm, y[0]*mm, y[-1]*mm])
        ax.set(title=f"{title} wrapped on the roller (W)\nroller time step {time_index}/{sino.shape[1] - 1} (final iterate)", xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(im, ax=ax)

    axs[1, 2].axis("off")
    fig.tight_layout()

    logs.show() #calls plt.show(), which also displays the figure above
    return fig, axs
