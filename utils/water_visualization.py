
import numpy as np
import matplotlib.pyplot as plt


def create_ndwi_figure(ndwi, threshold=0.0):
    """
    Create NDWI and potential-water visualization maps.

    NDWI values typically range from -1 to +1.
    Values above the threshold are highlighted as potential water,
    not confirmed floodwater.
    """
    ndwi = np.asarray(ndwi, dtype=np.float32)

    if ndwi.ndim != 2:
        raise ValueError("NDWI must be a two-dimensional array.")

    valid = np.isfinite(ndwi)

    if not valid.any():
        raise ValueError("NDWI contains no valid pixels to display.")

    # Mask invalid pixels so they appear transparent.
    ndwi_masked = np.ma.masked_invalid(ndwi)

    # Identify pixels classified as potential water.
    water_mask = valid & (ndwi > threshold)
    water_display = np.ma.masked_where(~valid, water_mask.astype(float))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        constrained_layout=True,
    )

    # Panel 1: Continuous NDWI index.
    index_image = axes[0].imshow(
        ndwi_masked,
        cmap="RdYlBu",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )

    axes[0].set_title("NDWI Index", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Pixel column")
    axes[0].set_ylabel("Pixel row")

    colorbar = fig.colorbar(
        index_image,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("NDWI value")

    # Panel 2: Binary potential-water classification.
    water_image = axes[1].imshow(
        water_display,
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    axes[1].set_title(
        f"Potential Water (NDWI > {threshold:.2f})",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("Pixel column")
    axes[1].set_ylabel("Pixel row")

    water_colorbar = fig.colorbar(
        water_image,
        ax=axes[1],
        ticks=[0, 1],
        fraction=0.046,
        pad=0.04,
    )
    water_colorbar.ax.set_yticklabels(["Not classified", "Potential water"])

    fig.suptitle(
        "TerraResQ AI — Satellite Water Analysis",
        fontsize=15,
        fontweight="bold",
    )

    return fig
