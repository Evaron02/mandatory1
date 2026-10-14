from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from Wave2D import Wave2D_Neumann


sol = Wave2D_Neumann()
data = sol(
    N=40,
    Nt=40,
    cfl=1 / np.sqrt(2),
    mx=2,
    my=2,
    store_data=1,
)

# These settings cover one complete oscillation.
steps = sorted(data)

fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
image = ax.imshow(
    data[0].T,
    origin="lower",
    extent=(0, 1, 0, 1),
    vmin=-1,
    vmax=1,
    cmap="RdBu_r",
    interpolation="nearest",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
title = ax.set_title("Neumann wave: t = 0.000")
fig.colorbar(image, ax=ax, label="u")
fig.tight_layout()


def update(n):
    image.set_data(data[n].T)
    title.set_text(f"Neumann wave: t = {n * sol.dt:.3f}")
    return image, title


animation = FuncAnimation(
    fig,
    update,
    frames=steps,
    interval=100,
    blit=False,
)

output = Path(__file__).resolve().parent / "report" / "neumannwave.gif"
output.parent.mkdir(exist_ok=True)
animation.save(output, writer=PillowWriter(fps=10))
plt.close(fig)

size = output.stat().st_size
print(f"Saved {output.name}: {size / 1000:.1f} kB")
assert size < 1_000_000, "The GIF must be smaller than 1 MB."