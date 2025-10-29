# GeoMCP - Satellite Data Analysis API Server

FastAPI-based geospatial analysis platform for processing real-time satellite imagery from multiple sources (Sentinel-1/2/3/5P, Landsat, MODIS, SMAP). Delivers 25+ REST endpoints for terrain and vegetation analysis with multi-format outputs.

## Overview

GeoMCP provides a comprehensive API for satellite data analysis, enabling environmental monitoring, climate research, agricultural management, and disaster response. The platform processes data from ESA's Sentinel Hub and other satellite sources, delivering results in multiple formats (PNG visualizations, GeoTIFF scientific data, JSON matrices).

### Use Cases

- **Environmental monitoring & climate research:** Track vegetation health, deforestation, wildfires, drought conditions, snow/ice cover, glacier movement, land subsidence, sea level rise, and atmospheric pollution
- **Agricultural & water resource management:** Monitor crop health/yields, soil moisture, irrigation needs, surface water extent, flood inundation, and precipitation patterns
- **Urban planning & disaster response:** Analyze urban expansion, infrastructure damage, active fires, land cover change, coastal erosion, and post-disaster impacts

## Current Features

### Active Modules

- **NDVI (Normalized Difference Vegetation Index):** Vegetation health and greenness monitoring using Sentinel-2 red and NIR bands
- **NDWI (Normalized Difference Water Index):** Water body detection and vegetation moisture content analysis
- **DEM Terrain Analysis:** Elevation mapping with derivatives including slope, aspect, hillshade, and flow accumulation
- **Hydrological Flow Modeling:** D8 flow direction and accumulation algorithms for watershed analysis

### Output Formats

Each analysis module supports three output formats:
- **PNG:** Visualizations for immediate viewing (grayscale or color-mapped)
- **GeoTIFF:** Scientific-grade float32 raster data for GIS applications
- **JSON Matrix:** Numerical arrays for programmatic analysis and LLM integration

## Installation

### Prerequisites

- Python 3.8+
- Sentinel Hub account with API credentials

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd GeoMCP
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API credentials:

Create `config/api_keys.env` with your Sentinel Hub credentials:
```env
SENTINELHUB_CLIENT_ID=your_client_id_here
SENTINELHUB_CLIENT_SECRET=your_client_secret_here
# OR use a static token:
# SENTINELHUB_TOKEN=your_token_here
```

5. Run the server:
```bash
uvicorn server:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- **Interactive API docs:** `http://localhost:8000/docs`
- **Alternative docs:** `http://localhost:8000/redoc`

### Key Endpoints

#### Health & Status
- `GET /health` - Server health check
- `GET /status` - Sentinel Hub token status

#### NDVI (Vegetation Analysis)
- `GET /ndvi.png` - NDVI visualization (grayscale PNG)
- `GET /analyze` - NDVI statistics (mean, median, percentiles)

**Parameters:**
- `bbox`: Bounding box as `minLon,minLat,maxLon,maxLat` (EPSG:4326)
- `from`: Start date (ISO format: `2024-01-01`)
- `to`: End date (ISO format: `2024-01-31`)
- `width`: Image width in pixels (default: 512)
- `height`: Image height in pixels (default: 512)

#### Elevation & Terrain
- `GET /elevation.png` - Grayscale elevation map
- `GET /elevation.matrix` - JSON elevation matrix
- `GET /elevation/raw` - Raw float32 GeoTIFF

**Derived Products:**
- `GET /hillshade.png` / `.tif` / `.matrix` - Illumination/shading visualization
- `GET /aspect.tif` / `.matrix` - Slope direction (0-360° from North)
- `GET /elevation/vectors.png` / `.tif` / `.matrix` - Slope vector field

#### Hydrological Analysis
- `GET /flow/accumulation.png` / `.tif` / `.matrix` - Flow accumulation (log-scaled)

#### Water Detection
- `GET /ndwi.png` / `.tif` / `.matrix` - Water body detection

### Example Requests

```bash
# Get NDVI for San Francisco Bay Area (Jan 2024)
curl "http://localhost:8000/ndvi.png?bbox=-122.5,37.5,-122.0,38.0&from=2024-01-01&to=2024-01-31" \
  --output ndvi.png

# Get elevation data as JSON matrix
curl "http://localhost:8000/elevation.matrix?bbox=-122.5,37.5,-122.0,38.0&width=256&height=256"

# Get hillshade with custom lighting
curl "http://localhost:8000/hillshade.png?bbox=-122.5,37.5,-122.0,38.0&azimuth_deg=315&altitude_deg=45" \
  --output hillshade.png

# Analyze NDVI statistics
curl "http://localhost:8000/analyze?bbox=-122.5,37.5,-122.0,38.0&from=2024-01-01&to=2024-01-31"
```

## Project Structure

```
GeoMCP/
├── server.py                  # FastAPI application & endpoints
├── config/
│   ├── api_keys.env          # API credentials (gitignored)
│   └── settings.py           # Configuration settings
├── modules/
│   ├── sentinel_hub.py       # Sentinel Hub API integration & OAuth
│   ├── elevation.py          # DEM analysis & terrain derivatives
│   ├── ndwi.py              # Water index calculations
│   ├── gpt_analysis.py      # Image analysis utilities
│   └── status.py            # Token status monitoring
├── Documentation/
│   └── Future Layers.md     # Roadmap for 70+ planned functions
└── .gitignore               # Excludes config/ and sensitive files
```

## Roadmap

The project has a documented expansion plan for **70+ additional satellite analysis functions** across multiple domains:

### Vegetation & Land Monitoring (15 functions)
EVI, SAVI, MSAVI, NDRE, NDMI, NBR, LAI, land cover classification, change detection, phenology, forest monitoring, crop health, landscape metrics, drought indices

### Water & Ocean Analysis (11 functions)
Flood mapping (SAR), wetlands, soil moisture (SMAP/Sentinel-1), ocean color (OLCI), SST, bathymetry, turbidity, coastal erosion, coral reefs, precipitation (GPM)

### Urban & Infrastructure (6 functions)
NDBI, urban expansion, nighttime lights (VIIRS), Local Climate Zones, damage assessment, object detection (sub-meter)

### Terrain & Geohazards (12 functions)
InSAR DEM, subsidence/uplift, glacier velocity, permafrost, feature tracking, 3D photogrammetry, evapotranspiration, LiDAR (GEDI)

### Climate & Atmosphere (10 functions)
Greenhouse gases (CO₂/CH₄), air quality (NO₂/SO₂), AOD, LST/albedo, sea level (altimetry), GRACE groundwater, volcanic monitoring

### Specialized Analysis (16 functions)
Snow/ice (NDSI), fire detection, sea ice, ship/oil detection, sensor fusion, polarimetric SAR, hyperspectral minerals, pan-sharpening, historical imagery

See `Documentation/Future Layers.md` for complete details on all planned functions.

## Technology Stack

- **Framework:** FastAPI (async Python web framework)
- **Geospatial:** NumPy, Rasterio, Tifffile, PIL/Pillow
- **Visualization:** Matplotlib
- **Data Sources:** Sentinel Hub API (Sentinel-1/2/3, SRTM DEM)
- **Authentication:** OAuth2 with token caching

## Configuration

### Environment Variables

Configure in `config/api_keys.env`:

```env
# Option 1: OAuth credentials (recommended)
SENTINELHUB_CLIENT_ID=your_client_id
SENTINELHUB_CLIENT_SECRET=your_client_secret

# Option 2: Static token
SENTINELHUB_TOKEN=your_static_token

# Optional: Custom base URL
SENTINELHUB_BASE_URL=https://services.sentinel-hub.com
```

### Resolution Guidelines

- **Quick analysis:** 256×256 or 512×512 pixels
- **Detailed analysis:** 1024×1024 pixels
- **High-resolution:** 2048×2048+ pixels (longer processing time)

### Bounding Box Format

All endpoints use EPSG:4326 (WGS84) coordinates:
```
bbox=minLongitude,minLatitude,maxLongitude,maxLatitude
```

Example (San Francisco): `bbox=-122.5,37.5,-122.0,38.0`

## Development

### Adding New Analysis Functions

1. Create a new module in `modules/` (e.g., `modules/new_analysis.py`)
2. Implement three output functions:
   - `get_<analysis>()` - PNG visualization
   - `get_<analysis>_raw()` - GeoTIFF float32
   - `get_<analysis>_matrix()` - JSON matrix
3. Add endpoints in `server.py` following the existing pattern
4. Update this README with the new functionality

### Running Tests

```bash
# Start the development server
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
```


## Acknowledgments

- ESA Copernicus Programme for Sentinel satellite data
- Sentinel Hub for API access and processing infrastructure
- NASA/USGS for SRTM elevation data

## Contact

edin.com/in/guillaume-atencia-2786b8252/
