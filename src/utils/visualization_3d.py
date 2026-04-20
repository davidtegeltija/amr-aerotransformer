from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_prediction(
    geom: np.ndarray,
    prediction: np.ndarray,
    *,
    title: str = "Adaptive Mesh",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:


    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(projection="3d")

    elev = 68; azim =120 

    colors, sm = color_map(prediction[..., 0], "gist_rainbow", alpha=1, dmin=-1, dmax=1)    # cp
    x = geom[:, :, 0]
    y = geom[:, :, 1]
    z = geom[:, :, 2]
    ax.plot_surface(x, y, z, facecolors=colors, edgecolor="none", rstride=1, cstride=3, shade=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M")
        fig.savefig(f"{save_path}/3d_visualisation-{timestamp}.png", bbox_inches="tight")
        print(f"[visualization] Saved → {save_path}")

    if show:
        plt.show()


def color_map(
    values: np.ndarray,
    cmap_name: str,
    *,
    alpha: float = 1.0,
    dmin: Optional[float] = None,
    dmax: Optional[float] = None,
):
    norm = mcolors.Normalize(
        vmin=dmin if dmin is not None else values.min(),
        vmax=dmax if dmax is not None else values.max(),
    )
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm(values))
    colors[..., 3] = alpha
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return colors, sm