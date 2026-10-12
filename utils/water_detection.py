
import numpy as np
import rasterio
from rasterio.enums import Resampling


def calculate_ndwi_from_urls(green_url, nir_url, max_size=1024):
    """Calculate NDWI from Sentinel-2 B03 and B08 cloud-hosted GeoTIFFs."""

    with rasterio.open(green_url) as green_src, rasterio.open(nir_url) as nir_src:
        scale = min(
            1.0,
            max_size / max(green_src.width, green_src.height),
        )

        height = max(1, int(green_src.height * scale))
        width = max(1, int(green_src.width * scale))

        green = green_src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.bilinear,
        ).astype("float32")

        nir = nir_src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.bilinear,
        ).astype("float32")

        green_nodata = green_src.nodata
        nir_nodata = nir_src.nodata

    if green.shape != nir.shape:
        raise ValueError("Green and NIR band dimensions do not match.")

    denominator = green + nir
    valid = np.isfinite(green) & np.isfinite(nir) & (denominator != 0)

    if green_nodata is not None:
        valid &= green != green_nodata
    if nir_nodata is not None:
        valid &= nir != nir_nodata

    ndwi = np.full(green.shape, np.nan, dtype="float32")
    ndwi[valid] = (
        (green[valid] - nir[valid]) / denominator[valid]
    )

    return ndwi


def classify_water(ndwi, threshold=0.0):
    """Return a mask of pixels that may represent water."""
    return np.isfinite(ndwi) & (ndwi > threshold)
