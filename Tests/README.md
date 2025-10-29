# GeoMCP Test Suite

Comprehensive test suite for the GeoMCP satellite data analysis API.

## Test Coverage

The test suite covers all API endpoints across multiple categories:

- **Health & Status** (`test_health.py`) - Health checks and system status
- **Elevation** (`test_elevation.py`) - DEM and elevation products (PNG, TIFF, matrix, gradients, vectors)
- **Indices** (`test_indices.py`) - NDVI, NDWI, NDBI spectral indices
- **Terrain** (`test_terrain.py`) - Slope, aspect, hillshade, flow accumulation
- **Cloud-free** (`test_cloudfree.py`) - Cloud-free composite imagery
- **Classification** (`test_classification.py`) - Rule-based, unsupervised, supervised classification
- **Zonal Analysis** (`test_zonal.py`) - Zonal statistics and time-series analysis

## Quick Start

### 1. Install Test Dependencies

```bash
pip install -r Tests/requirements-test.txt
```

### 2. Run All Tests

```bash
./run_tests.sh
```

## Usage

### Basic Test Execution

```bash
# Run all tests
./run_tests.sh

# Verbose output
./run_tests.sh -v

# Very verbose output
./run_tests.sh -vv
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
./run_tests.sh -p
```

### Coverage Reports

```bash
# Generate coverage report
./run_tests.sh -c

# View coverage report
open htmlcov/index.html
```

### HTML Test Reports

```bash
# Generate HTML test report
./run_tests.sh --html

# View report
open Tests/report.html
```

### Running Specific Tests

```bash
# Run specific test file
./run_tests.sh -t test_health.py

# Run specific test class
./run_tests.sh -t test_health.py::TestHealth

# Run specific test function
./run_tests.sh -t test_health.py::test_health_endpoint
```

### Combining Options

```bash
# Verbose, parallel, with coverage and HTML report
./run_tests.sh -vv -p -c --html
```

## Test Structure

```
Tests/
├── conftest.py              # Shared fixtures and configuration
├── test_health.py           # Health and status endpoints
├── test_elevation.py        # Elevation and DEM tests
├── test_indices.py          # NDVI, NDWI, NDBI tests
├── test_terrain.py          # Terrain analysis tests
├── test_cloudfree.py        # Cloud-free composite tests
├── test_classification.py   # Classification tests
├── test_zonal.py            # Zonal stats and time-series tests
├── requirements-test.txt    # Test dependencies
└── README.md                # This file
```

## Fixtures

Common fixtures are defined in `conftest.py`:

- `client` - FastAPI TestClient
- `sample_bbox` - Sample bounding box string
- `sample_bbox_list` - Sample bounding box as list
- `sample_date_range` - Sample from/to dates
- `sample_geometry` - Sample GeoJSON polygon
- `small_dimensions` - Small image dimensions (64x64) for faster tests
- `sample_training_points` - Training points for supervised classification

## Configuration

### Environment Variables

Tests use the same environment configuration as the main application:
- `SENTINELHUB_CLIENT_ID`
- `SENTINELHUB_CLIENT_SECRET`
- `SENTINELHUB_TOKEN`

Make sure your `config/api_keys.env` is properly configured.

### Test Dimensions

Most tests use small dimensions (64x64 or 32x32) to speed up execution. Real-world usage typically uses 512x512 or larger.

## Writing New Tests

### Example Test

```python
def test_new_endpoint(client, sample_bbox, small_dimensions):
    """Test description."""
    response = client.get(
        f"/new/endpoint?bbox={sample_bbox}&width={small_dimensions['width']}&height={small_dimensions['height']}"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
```

### Test Naming Convention

- Test files: `test_<category>.py`
- Test classes: `Test<Feature>`
- Test functions: `test_<specific_behavior>`

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
- name: Run tests
  run: |
    pip install -r Tests/requirements-test.txt
    ./run_tests.sh -p -c --html
```

## Troubleshooting

### Tests Failing with API Errors

- Check that Sentinel Hub credentials are configured in `config/api_keys.env`
- Verify your token hasn't expired: `curl http://localhost:8000/status`
- Some endpoints require valid satellite data for the requested bbox/dates

### Slow Tests

- Use parallel execution: `./run_tests.sh -p`
- Tests use small dimensions by default, but API calls still require network requests
- Consider mocking external API calls for unit tests vs. integration tests

### Import Errors

- Make sure you're running from the project root directory
- Virtual environment should be activated
- All dependencies installed: `pip install -r requirements.txt`

## Contributing

When adding new endpoints:

1. Add corresponding tests to the appropriate test file
2. Use existing fixtures where possible
3. Test both success and error cases
4. Verify response format and content types
5. Run full test suite before committing: `./run_tests.sh -c`

## License

Same as main project.
