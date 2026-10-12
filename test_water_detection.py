
import numpy as np

from utils.satellite_data import search_satellite_images
from utils.water_detection import calculate_ndwi_from_urls, classify_water

# Search for recent Sentinel-2 imagery over Kerala.
scenes = search_satellite_images(
    [76.0, 9.5, 76.5, 10.0],
    days_back=30,
    max_items=5,
)

if not scenes:
    print("No satellite scenes found. Try a wider date range.")
else:
    scene = scenes[0]

    if not scene["green_url"] or not scene["nir_url"]:
        print("The selected scene is missing B03 or B08.")
    else:
        print("Scene date:", scene["date"])
        print("Cloud cover:", scene["cloud_cover"])

        ndwi = calculate_ndwi_from_urls(
            scene["green_url"],
            scene["nir_url"],
        )

        water_mask = classify_water(ndwi)
        valid_pixels = np.isfinite(ndwi)

        if valid_pixels.any():
            water_percent = (
                water_mask[valid_pixels].mean() * 100
            )
            print("NDWI calculation successful!")
            print("Valid pixels:", int(valid_pixels.sum()))
            print(f"Potential water pixels: {water_percent:.2f}%")
        else:
            print("No valid pixels found.")
