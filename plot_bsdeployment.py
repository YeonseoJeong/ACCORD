from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from env.env import ORANLoadBalancingEnv

env = ORANLoadBalancingEnv(Path(__file__).with_name("config.yaml"))

centers = env.cell_xy
R = env.hex_radius
isd = env.isd

fig, ax = plt.subplots(figsize=(9, 8))

angles = np.pi / 2 + np.arange(6) * np.pi / 3

unit_vertices = np.column_stack([
    np.cos(angles),
    np.sin(angles)
])


for b, center in enumerate(centers):

    vertices = center + R * unit_vertices

    hexagon = Polygon(
        vertices,
        closed=True,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.13
    )

    ax.add_patch(hexagon)

    # BS position
    ax.plot(
        center[0],
        center[1],
        "o",
        color="tab:red",
        markersize=8
    )

    ax.annotate(
        f"BS {b}",
        xy=(center[0], center[1]),
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=11,
        weight="bold"
    )


bs0 = centers[0]
bs1 = centers[1]

ax.annotate(
    "",
    xy=(bs1[0], bs1[1]),
    xytext=(bs0[0], bs0[1]),
    arrowprops=dict(arrowstyle="<->", lw=2, color="tab:green")
)

mid_isd = (bs0 + bs1) / 2
ax.text(
    mid_isd[0],
    mid_isd[1] - 10,
    f"ISD = {isd:.0f} m",
    color="tab:green",
    fontsize=11,
    ha="center",
    va="top",
    weight="bold"
)

# top vertex of BS0 hexagon
vertex0 = bs0 + R * unit_vertices[0]

ax.annotate(
    "",
    xy=(vertex0[0], vertex0[1]),
    xytext=(bs0[0], bs0[1]),
    arrowprops=dict(arrowstyle="<->", lw=2, color="tab:purple")
)

mid_r = (bs0 + vertex0) / 2
ax.text(
    mid_r[0] + 12,
    mid_r[1],
    f"R = {R:.2f} m",
    color="tab:purple",
    fontsize=11,
    ha="left",
    va="center",
    weight="bold"
)


# Plot settings
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("7-cell BS Deployment")

ax.grid(True, linestyle=":", alpha=0.35)

ax.autoscale_view()
plt.tight_layout()

save_path = Path(__file__).resolve().parent / "bs_deployment.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Figure saved to {save_path}")