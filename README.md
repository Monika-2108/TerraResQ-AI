# 🌧️ TerraResQ AI
### Satellite-Based Flood Change Screening

**TerraResQ AI** is an AI-assisted disaster-response prototype that analyzes Sentinel-2 satellite imagery to identify potential water changes that may help prioritize flood assessment.

## Problem
Flood response teams need timely information about changing water coverage to help identify areas that may require further investigation.

## Our Solution
TerraResQ AI provides:
- 🛰️ Satellite imagery discovery using Microsoft Planetary Computer.
- 🌊 Normalized Difference Water Index (NDWI) analysis.
- 🔍 Before-and-after satellite observation comparison.
- 🗺️ Visualization of potential newly detected water.
- 📊 Comparison statistics for valid pixels and potential water changes.

## Technology Stack
- Python
- Streamlit
- Sentinel-2 satellite imagery
- Microsoft Planetary Computer STAC API
- Rasterio and NumPy
- Matplotlib and Pandas

## How It Works
1. Select a geographic region.
2. Search for available Sentinel-2 observations.
3. Select an observation to analyze potential water coverage.
4. Select two dates from the same satellite tile.
5. Align the later observation to the earlier observation's spatial grid.
6. Compare NDWI classifications and visualize potential newly detected water.

## Important Limitations
This is a prototype for flood-change screening, not a verified flood-detection or emergency-response system. NDWI changes can also result from seasonal water, shadows, clouds, or other environmental differences. Independent validation is required.

Road and bridge damage detection, identification of isolated communities, and verified rescue-route generation are not implemented.

## Run Locally
```bash
python -m streamlit run app.py
```

## Project Goal
Support faster preliminary assessment of potential flood-related water changes using accessible satellite data.
