
import re
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ReadResampling
from rasterio.transform import array_bounds

from utils.satellite_data import search_satellite_images


st.set_page_config(
    page_title="TerraResQ AI",
    page_icon="🌧️",
    layout="wide",
)

st.title("🌧️ TerraResQ AI")
st.caption(
    "Satellite-based water analysis and potential flood-change screening"
)

st.warning(
    "Research prototype: detected water changes are not verified floods. "
    "Clouds, shadows, seasonal water, and image-processing differences can "
    "affect results. This tool does not generate verified rescue routes."
)


REGIONS = {
    "Kerala": [74.85, 8.18, 77.40, 12.80],
    "Assam": [89.60, 24.10, 96.00, 28.20],
    "Bengaluru": [77.35, 12.75, 77.85, 13.20],
}


def get_tile_code(scene):
    """Extract the Sentinel-2 MGRS tile code when present."""
    match = re.search(r"T\d{2}[A-Z]{3}", scene.get("item_id", ""))
    return match.group(0) if match else None


def read_band(url, max_size=1024):
    """Read a band at reduced resolution and retain its spatial metadata."""
    with rasterio.open(url) as src:
        scale = min(1.0, max_size / max(src.width, src.height))
        height = max(1, round(src.height * scale))
        width = max(1, round(src.width * scale))

        data = src.read(
            1,
            out_shape=(height, width),
            resampling=ReadResampling.bilinear,
        ).astype("float32")

        transform = src.transform * src.transform.scale(
            src.width / width,
            src.height / height,
        )

        return {
            "data": data,
            "transform": transform,
            "crs": src.crs,
            "nodata": src.nodata,
            "bounds": src.bounds,
        }


def read_scene_ndwi(scene, reference=None, max_size=1024):
    """
    Calculate NDWI after aligning green and NIR bands to one grid.

    If reference is supplied, align this scene to the reference scene's
    CRS, transform, and dimensions.
    """
    green_url = scene.get("green_url")
    nir_url = scene.get("nir_url")

    if not green_url or not nir_url:
        raise ValueError("This scene does not contain green and NIR bands.")

    green_source = read_band(green_url, max_size)
    nir_source = read_band(nir_url, max_size)

    if green_source["crs"] is None or nir_source["crs"] is None:
        raise ValueError("A satellite band is missing its coordinate system.")

    if reference is None:
        destination_crs = green_source["crs"]
        destination_transform = green_source["transform"]
        destination_height, destination_width = green_source["data"].shape
    else:
        destination_crs = reference["crs"]
        destination_transform = reference["transform"]
        destination_height, destination_width = reference["ndwi"].shape

    def warp_band(source):
        destination = np.full(
            (destination_height, destination_width),
            np.nan,
            dtype="float32",
        )

        reproject(
            source=source["data"],
            destination=destination,
            src_transform=source["transform"],
            src_crs=source["crs"],
            src_nodata=source["nodata"],
            dst_transform=destination_transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )
        return destination

    green = warp_band(green_source)
    nir = warp_band(nir_source)

    denominator = green + nir
    valid = (
        np.isfinite(green)
        & np.isfinite(nir)
        & (np.abs(denominator) > 1e-8)
    )

    ndwi = np.full(green.shape, np.nan, dtype="float32")
    ndwi[valid] = (green[valid] - nir[valid]) / denominator[valid]

    return {
        "ndwi": ndwi,
        "crs": destination_crs,
        "transform": destination_transform,
    }


def compare_scenes(before_scene, after_scene, threshold=0.0):
    """
    Align the later scene to the earlier scene's grid and compare NDWI.
    Only overlapping pixels valid in both observations are counted.
    """
    before = read_scene_ndwi(before_scene)
    after = read_scene_ndwi(after_scene, reference=before)

    before_ndwi = before["ndwi"]
    after_ndwi = after["ndwi"]

    valid = np.isfinite(before_ndwi) & np.isfinite(after_ndwi)

    before_water = valid & (before_ndwi > threshold)
    after_water = valid & (after_ndwi > threshold)
    newly_detected = valid & ~before_water & after_water

    valid_count = int(valid.sum())
    new_count = int(newly_detected.sum())
    new_percent = 100.0 * new_count / valid_count if valid_count else 0.0

    return {
        "before": before_ndwi,
        "after": after_ndwi,
        "newly_detected": newly_detected,
        "valid_pixels": valid_count,
        "new_pixels": new_count,
        "new_percent": new_percent,
        "crs": before["crs"],
        "transform": before["transform"],
    }


def show_comparison(result, before_date, after_date):
    """Display the two NDWI maps and potential newly detected water."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    panels = [
        (result["before"], "Before: NDWI", "RdYlBu", -1, 1),
        (result["after"], "After: NDWI (aligned)", "RdYlBu", -1, 1),
        (
            np.ma.masked_where(
                ~np.isfinite(result["before"]) | ~np.isfinite(result["after"]),
                result["newly_detected"].astype(float),
            ),
            "Potential New Water",
            "Blues",
            0,
            1,
        ),
    ]

    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(
            np.ma.masked_invalid(data),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Pixel column")
        ax.set_ylabel("Pixel row")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"TerraResQ AI | {before_date} → {after_date}",
        fontsize=14,
        fontweight="bold",
    )
    st.pyplot(fig)
    plt.close(fig)


with st.sidebar:
    st.header("Analysis settings")
    region = st.selectbox("Select region", list(REGIONS.keys()))
    days_back = st.slider("Search period (days)", 15, 120, 60, step=15)
    max_items = st.slider("Maximum satellite scenes", 5, 30, 20)
    threshold = st.slider(
        "Potential-water NDWI threshold",
        min_value=-0.2,
        max_value=0.4,
        value=0.0,
        step=0.05,
    )

st.subheader("🛰️ Satellite catalogue")

if st.button("Search Satellite Images", type="primary"):
    try:
        with st.spinner("Searching Sentinel-2 observations..."):
            scenes = search_satellite_images(
                REGIONS[region],
                days_back=days_back,
                max_items=max_items,
            )
        st.session_state["terra_scenes"] = scenes
        st.session_state["terra_region"] = region
    except Exception as exc:
        st.error(f"Satellite search failed: {exc}")

scenes = st.session_state.get("terra_scenes", [])

if not scenes:
    st.info("Choose a region and click Search Satellite Images to begin.")
    st.stop()

st.success(f"Satellite catalogue connected — {len(scenes)} scenes found.")

table_rows = [
    {
        "Date": scene.get("date", "Unknown"),
        "Cloud cover (%)": scene.get("cloud_cover"),
        "Tile": get_tile_code(scene) or "Unknown",
        "Scene ID": scene.get("item_id", ""),
    }
    for scene in scenes
]
st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

st.subheader("🔎 Individual scene analysis")

usable_scenes = [
    scene for scene in scenes
    if scene.get("green_url") and scene.get("nir_url")
]

if not usable_scenes:
    st.warning("No scenes with usable green and NIR bands were found.")
    st.stop()

scene_labels = {
    f'{scene.get("date", "Unknown")} | '
    f'{get_tile_code(scene) or "Unknown tile"} | '
    f'{scene.get("item_id", "")}': scene
    for scene in usable_scenes
}

selected_label = st.selectbox(
    "Select an observation",
    list(scene_labels.keys()),
)
selected_scene = scene_labels[selected_label]

preview_url = selected_scene.get("preview_url") or selected_scene.get("visual_url")
if preview_url:
    st.image(preview_url, caption=f"Sentinel-2 preview — {selected_scene.get('date')}")

if st.button("Analyze Selected Scene"):
    try:
        with st.spinner("Calculating NDWI..."):
            result = read_scene_ndwi(selected_scene)
        water = np.isfinite(result["ndwi"]) & (result["ndwi"] > threshold)
        valid_count = int(np.isfinite(result["ndwi"]).sum())
        water_count = int(water.sum())
        percentage = 100 * water_count / valid_count if valid_count else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Valid pixels", f"{valid_count:,}")
        col2.metric("Potential-water pixels", f"{water_count:,}")
        col3.metric("Potential-water share", f"{percentage:.2f}%")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        ndwi_plot = axes[0].imshow(
            np.ma.masked_invalid(result["ndwi"]),
            cmap="RdYlBu",
            vmin=-1,
            vmax=1,
        )
        axes[0].set_title("NDWI Index")
        fig.colorbar(ndwi_plot, ax=axes[0], fraction=0.046, pad=0.04)

        water_plot = axes[1].imshow(
            np.ma.masked_where(
                ~np.isfinite(result["ndwi"]),
                water.astype(float),
            ),
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        axes[1].set_title(f"Potential Water (NDWI > {threshold:.2f})")
        fig.colorbar(water_plot, ax=axes[1], fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "NDWI classifies potential water, not confirmed floodwater. "
            "The percentage describes valid pixels in this scene."
        )
    except Exception as exc:
        st.error(f"NDWI analysis failed: {exc}")

st.divider()
st.subheader("🌧️ Before-and-After Water Comparison")

tile_scenes = {}
for scene in usable_scenes:
    tile = get_tile_code(scene)
    if tile:
        tile_scenes.setdefault(tile, []).append(scene)

if not tile_scenes:
    st.warning("No usable MGRS tile codes were found in the scene IDs.")
    st.info("Try another region or a longer search period.")
    st.stop()

selected_tile = st.selectbox("Choose a shared Sentinel-2 tile", sorted(tile_scenes))

same_tile = sorted(
    tile_scenes[selected_tile],
    key=lambda scene: scene.get("date", ""),
)

if len(same_tile) < 2:
    st.warning(
        "At least two observations of the same tile are required. "
        "Increase the search period and search again."
    )
    st.stop()

date_labels = {
    f'{scene.get("date", "Unknown")} | {scene.get("item_id", "")}': scene
    for scene in same_tile
}

before_label = st.selectbox(
    "Earlier observation",
    list(date_labels.keys()),
    index=0,
)
after_default = min(1, len(date_labels) - 1)
after_label = st.selectbox(
    "Later observation",
    list(date_labels.keys()),
    index=after_default,
)

before_scene = date_labels[before_label]
after_scene = date_labels[after_label]

if before_scene["item_id"] == after_scene["item_id"]:
    st.warning("Select two different satellite observations.")
elif before_scene.get("date", "") >= after_scene.get("date", ""):
    st.warning("Choose an earlier observation first and a later observation second.")
else:
    st.caption(
        "The later observation is reprojected to the earlier observation's grid. "
        "Only pixels valid in both scenes are included."
    )

    if st.button("Compare Water Changes", type="primary"):
        try:
            with st.spinner("Aligning satellite grids and comparing NDWI..."):
                comparison = compare_scenes(
                    before_scene,
                    after_scene,
                    threshold=threshold,
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Valid comparison pixels", f'{comparison["valid_pixels"]:,}')
            c2.metric("New potential-water pixels", f'{comparison["new_pixels"]:,}')
            c3.metric("New potential-water share", f'{comparison["new_percent"]:.2f}%')

            show_comparison(
                comparison,
                before_scene.get("date", "Unknown"),
                after_scene.get("date", "Unknown"),
            )

            st.caption(
                f"Comparison grid CRS: {comparison['crs']}. "
                "This is a screening result, not a verified flood map. "
                "Validate against independent flood reports and inspect the imagery "
                "before making emergency decisions."
            )
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

st.divider()
st.subheader("🚨 Project Status")
st.write("- Satellite catalogue: Connected")
st.write("- NDWI analysis: Available")
st.write("- Before/after grid alignment: Applied in comparison")
st.write("- Flood verification: Not implemented")
st.write("- Road/bridge damage and rescue-route verification: Not implemented")
