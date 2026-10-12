
from datetime import datetime, timedelta, timezone

import planetary_computer
import pystac_client


def search_satellite_images(bbox, days_back=30, max_items=10):
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=(
            f"{start_date:%Y-%m-%d}/"
            f"{end_date:%Y-%m-%d}"
        ),
        query={"eo:cloud_cover": {"lt": 30}},
        max_items=max_items,
    )

    results = []

    for item in search.items():
        preview = item.assets.get("rendered_preview")
        visual = item.assets.get("visual")
        green = item.assets.get("B03")
        nir = item.assets.get("B08")

        results.append({
            "date": (
                item.datetime.strftime("%Y-%m-%d")
                if item.datetime else "Unknown"
            ),
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "item_id": item.id,
            "collection": item.collection_id,
            "assets": list(item.assets.keys()),
            "preview_url": preview.href if preview else None,
            "visual_url": visual.href if visual else None,
            "green_url": green.href if green else None,
            "nir_url": nir.href if nir else None,
        })

    return results
