
import numpy as np
import matplotlib.pyplot as plt


def create_flood_comparison_figure(
    before_ndwi,
    after_ndwi,
    comparison,
):
    """Visualize before/after NDWI and potential newly detected water."""

    before = np.asarray(before_ndwi, dtype=np.float32)
    after = np.asarray(after_ndwi, dtype=np.float32)
    new_water = np.asarray(
        comparison["newly_detected_water"],
        dtype=bool,
    )

    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("NDWI maps must be two-dimensional.")

    if before.shape != after.shape or before.shape != new_water.shape:
        raise ValueError("All maps must have matching dimensions.")

    before_display = np.ma.masked_invalid(before)
    after_display = np.ma.masked_invalid(after)

    valid = np.isfinite(before) & np.isfinite(after)
    new_water_display = np.ma.masked_where(
        ~valid,
        new_water.astype(np.float32),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        constrained_layout=True,
    )

    first = axes[0].imshow(
        before_display,
        cmap="RdYlBu",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axes[0].set_title("Before: NDWI")
    fig.colorbar(first, ax=axes[0], fraction=0.046, pad=0.04)

    second = axes[1].imshow(
        after_display,
        cmap="RdYlBu",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_title("After: NDWI")
    fig.colorbar(second, ax=axes[1], fraction=0.046, pad=0.04)

    third = axes[2].imshow(
        new_water_display,
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[2].set_title("Potential Newly Detected Water")
    fig.colorbar(
        third,
        ax=axes[2],
        ticks=[0, 1],
        fraction=0.046,
        pad=0.04,
    )

    for axis in axes:
        axis.set_xlabel("Pixel column")
        axis.set_ylabel("Pixel row")

    fig.suptitle(
        "TerraResQ AI — Before/After Water Comparison",
        fontsize=15,
        fontweight="bold",
    )

    return fig
