
import numpy as np


def compare_water_maps(before_ndwi, after_ndwi, threshold=0.0):
    """
    Compare NDWI arrays from before and after satellite observations.

    Returns masks for potential water before, after, and newly detected
    water. Both arrays must cover the same area at matching resolution.
    """
    before = np.asarray(before_ndwi, dtype=np.float32)
    after = np.asarray(after_ndwi, dtype=np.float32)

    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("Both NDWI inputs must be two-dimensional arrays.")

    if before.shape != after.shape:
        raise ValueError(
            "Before and after NDWI maps must have matching dimensions."
        )

    valid = np.isfinite(before) & np.isfinite(after)

    water_before = valid & (before > threshold)
    water_after = valid & (after > threshold)

    # Pixels changing from non-water to potential water.
    newly_detected_water = valid & ~water_before & water_after

    valid_count = int(valid.sum())
    new_water_count = int(newly_detected_water.sum())

    new_water_percent = (
        (new_water_count / valid_count) * 100
        if valid_count > 0
        else 0.0
    )

    return {
        "water_before": water_before,
        "water_after": water_after,
        "newly_detected_water": newly_detected_water,
        "valid_pixels": valid_count,
        "new_water_pixels": new_water_count,
        "new_water_percent": new_water_percent,
    }
